import os
import time
import threading
import queue
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from map_closures import map_closures
from tqdm import tqdm, trange
from kiss_icp.kiss_icp import KissICP
from kiss_icp.voxelization import voxel_down_sample

from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2 
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs_py import point_cloud2 as pc2

from nav_msgs.msg import Path, OccupancyGrid
import tf2_ros
import numpy as np
from kiss_slam.slam import KissSLAM, transform_points
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.voxel_map import VoxelMap
from kiss_slam_ros.utils.config import declare_parameters, get_kiss_slam_config


# setup rclpy logging
rclpy.logging.set_logger_level('kiss_slam_ros', rclpy.logging.LoggingSeverity.INFO)

# Data structure for thread communication
class ScanData:
    def __init__(self, points, timestamps, current_pose, deskewed_frame):
        self.points = points.copy()  # Ensure points are copied to avoid threading issues
        self.timestamps = timestamps.copy()  # Copy timestamps to avoid threading issues
        self.current_pose = current_pose.copy()  # Copy current pose to avoid threading issues
        self.deskewed_frame = deskewed_frame.copy()  # Copy deskewed frame to avoid threading issues

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # Declare parameters
        declare_parameters(self)
        
        # Get parameters
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.loop_closure_threshold = self.get_parameter('loop_closure_threshold').get_parameter_value().integer_value
        self.optimization_frequency = self.get_parameter('optimization_frequency').get_parameter_value().double_value

        # Build KISS-SLAM config
        config = get_kiss_slam_config(self)
        
        # log all config parameters
        rclpy.logging.get_logger('kiss_slam_ros').info(
            f'KISS-SLAM configuration:\n'
            f'  Map Frame: {self.map_frame}\n'
            f'  Odometry Frame: {self.odom_frame}\n'
            f'  Base Frame: {self.base_frame}\n'
            f'  Loop Closure Threshold: {self.loop_closure_threshold}\n'
            f'  Optimization Frequency: {self.optimization_frequency}\n'
            f'  Odometry Config: {config.odometry}\n'
            f'  Local Mapper Config: {config.local_mapper}\n'
            f'  Occupancy Mapper Config: {config.occupancy_mapper}\n'
            f'  Loop Closer Config: {config.loop_closer}\n'
            f'  Pose Graph Optimizer Config: {config.pose_graph_optimizer}\n'
        )

        # Initialize core SLAM components
        self.odometry = KissICP(config.kiss_icp_config())
        self.slam = KissSLAM(config)
        self.occupancy_mapper = OccupancyGridMapper(config.occupancy_mapper)
        self.slam_config = config  # Store config for later use
        
        # Threading setup
        self.scan_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.slam_thread = None
        self.shutdown_event = threading.Event()
        
        # Thread-safe data sharing
        self._poses_lock = threading.Lock()
        self._current_keyposes = []
        self._current_poses = []
        
        # Start SLAM processing thread
        self.slam_thread = threading.Thread(target=self._slam_processing_thread, daemon=True)
        self.slam_thread.start()
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.slam_path_pub = self.create_publisher(Path, 'slam_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)
        self.local_pose_pub = self.create_publisher(PoseStamped, 'local_pose', 10)
        self.global_path_pub = self.create_publisher(Path, 'global_path', 10)
        self.global_pose_pub = self.create_publisher(PoseStamped, 'global_pose', 10)
        
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
    

        self.last_map_correction = np.eye(4)  # Last map->odom correction
        self.corrected_poses = []  # Store corrected poses after optimization
        self.pose_corrections_available = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            '/points_raw',
            self.listener_callback,
            qos_profile)
        
        self.odom_path_msg = Path()
        self.global_path_msg = Path()
        
        # Store the latest lidar message timestamp for map publishing
        self.latest_lidar_timestamp = None

    def listener_callback(self, msg):
        """Fast odometry thread - processes scans and publishes odom->base_link transform"""
        # Extract points and timestamps from PointCloud2 message
        points_np = pc2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True)
        timestamps = points_np['t'].astype(np.float64)
        points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)

        # Odometry processing (main thread - fast and real-time)
        deskewed_frame, _ = self.odometry.register_frame(points, timestamps)
        current_pose = self.odometry.last_pose
        
        # Prepare data for mapping (in background thread)
        scan_data = ScanData(
            points=points,
            timestamps=timestamps,
            current_pose=current_pose,
            deskewed_frame=deskewed_frame,
        )
        try:
            self.scan_queue.put(scan_data, timeout=1.0)  # Block until space is available
        except queue.Full:
            rclpy.logging.get_logger('kiss_slam_ros').warn(
                'Scan queue is full, dropping scan data'
            )
        # Store the latest lidar timestamp for map publishing
        self.latest_lidar_timestamp = msg.header.stamp

        # Create timestamp for this frame
        current_time = msg.header.stamp
        
        # Publish odom->base_link transform
        self._publish_odom_transform(current_pose, current_time)

        map_T_odom = self._current_keyposes[-1] if len(self._current_keyposes) > 0 else np.eye(4)
        # Publish map->odom transform (using aligned transformation)
        self._publish_map_transform(map_T_odom, current_time)
        
        # Publish odometry pose and path
        self._publish_odometry_pose_and_path(current_pose, current_time)

        # Publish global pose from map to base_link (using aligned transformation)
        self._publish_global_path(map_T_odom, current_pose, current_time)
        
        # Publish corrected SLAM path (using thread-safe poses)
        self._publish_slam_path(current_time)


    def _slam_processing_thread(self):
        """Background thread for SLAM processing (loop closure and optimization)"""
        log = rclpy.logging.get_logger('kiss_slam_ros')
        log.info('SLAM processing thread started')

        while not self.shutdown_event.is_set():
            try:
                scan = self.scan_queue.get(timeout=1.0)

                # 1) voxelize
                voxel_size = self.slam.local_map_voxel_size
                mapping_frame = voxel_down_sample(scan.deskewed_frame, voxel_size)

                # 2) traveled distance (from origin)
                traveled = np.linalg.norm(scan.current_pose[:3, -1])

                # 3) integrate into the map
                self.slam.voxel_grid.integrate_frame(mapping_frame, scan.current_pose)
                self.slam.local_map_graph.last_local_map.local_trajectory.append(scan.current_pose)

                # 4) if it’s time for a new node…
                poses = None
                if traveled > self.slam.local_map_splitting_distance:
                    # a) get points of the current local map
                    points = self.odometry.local_map.point_cloud()
                    # b) reset odometry
                    last_local_map = self.slam.local_map_graph.last_local_map
                    relative_motion = last_local_map.local_trajectory[-1]
                    inverse_relative_motion = np.linalg.inv(relative_motion)
                    transformed_local_map = transform_points(points, inverse_relative_motion)

                    self.odometry.local_map.clear()
                    self.odometry.local_map.add_points(transformed_local_map)
                    self.odometry.last_pose = np.eye(4)

                    # c) calculate closures
                    query_id = last_local_map.id
                    query_points = self.slam.voxel_grid.point_cloud()
                    self.slam.local_map_graph.finalize_local_map(self.slam.voxel_grid)
                    self.slam.voxel_grid.clear()
                    self.slam.voxel_grid.add_points(transformed_local_map)
                    self.slam.optimizer.add_variable(self.slam.local_map_graph.last_id, self.slam.local_map_graph.last_keypose)
                    self.slam.optimizer.add_factor(
                        self.slam.local_map_graph.last_id, query_id, relative_motion, np.eye(6)
                    )
                    self.slam.compute_closures(query_id, query_points)

                    poses, _ = self.slam.fine_grained_optimization()


                # Update thread-safe poses
                with self._poses_lock:
                    self._current_keyposes = self.slam.get_keyposes()
                    self._current_poses = poses if poses is not None else self.slam.poses

                self.scan_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f'Error in SLAM processing thread: {e}')
                continue

        log.info('SLAM processing thread stopped')




    def _publish_odom_transform(self, odom_pose, timestamp):
        """Publish odom->base_link transform"""
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = timestamp
        t_odom_base.header.frame_id = self.odom_frame
        t_odom_base.child_frame_id = self.base_frame
        t_odom_base.transform.translation.x = odom_pose[0, 3]
        t_odom_base.transform.translation.y = odom_pose[1, 3]
        t_odom_base.transform.translation.z = odom_pose[2, 3]
        q_odom_base = R.from_matrix(odom_pose[:3, :3]).as_quat()
        t_odom_base.transform.rotation.x = q_odom_base[0]
        t_odom_base.transform.rotation.y = q_odom_base[1]
        t_odom_base.transform.rotation.z = q_odom_base[2]
        t_odom_base.transform.rotation.w = q_odom_base[3]
        
        self.tf_broadcaster.sendTransform(t_odom_base)

    def _publish_map_transform(self, last_keypose, timestamp):
        """Publish map->odom transform using last local map keypose"""
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = timestamp
        t_map_odom.header.frame_id = self.map_frame
        t_map_odom.child_frame_id = self.odom_frame
        t_map_odom.transform.translation.x = last_keypose[0, 3]
        t_map_odom.transform.translation.y = last_keypose[1, 3]
        t_map_odom.transform.translation.z = last_keypose[2, 3]
        q_map_odom = R.from_matrix(last_keypose[:3, :3]).as_quat()
        t_map_odom.transform.rotation.x = q_map_odom[0]
        t_map_odom.transform.rotation.y = q_map_odom[1]
        t_map_odom.transform.rotation.z = q_map_odom[2]
        t_map_odom.transform.rotation.w = q_map_odom[3]
        
        self.tf_broadcaster.sendTransform(t_map_odom)

    def _publish_odometry_pose_and_path(self, odom_pose, timestamp):
        """Publish odometry pose and path"""
        # Publish odometry pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.odom_frame
        pose_msg.pose.position.x = odom_pose[0, 3]
        pose_msg.pose.position.y = odom_pose[1, 3]
        pose_msg.pose.position.z = odom_pose[2, 3]
        q_odom = R.from_matrix(odom_pose[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = q_odom[0]
        pose_msg.pose.orientation.y = q_odom[1]
        pose_msg.pose.orientation.z = q_odom[2]
        pose_msg.pose.orientation.w = q_odom[3]
        self.local_pose_pub.publish(pose_msg)

        # Publish odometry path
        self.odom_path_msg.header.stamp = timestamp
        self.odom_path_msg.header.frame_id = self.odom_frame
        self.odom_path_msg.poses.append(pose_msg)
        self.odom_path_pub.publish(self.odom_path_msg)
    
    def _publish_global_path(self, map_T_odom, odom_T_base, timestamp):
        """Publish global pose from map to base_link using the odom pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.map_frame
        map_T_base = map_T_odom @ odom_T_base
        pose_msg.pose.position.x = map_T_base[0, 3]
        pose_msg.pose.position.y = map_T_base[1, 3]
        pose_msg.pose.position.z = map_T_base[2, 3]
        q_map_base = R.from_matrix(map_T_base[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = q_map_base[0]
        pose_msg.pose.orientation.y = q_map_base[1]
        pose_msg.pose.orientation.z = q_map_base[2]
        pose_msg.pose.orientation.w = q_map_base[3]
        self.global_pose_pub.publish(pose_msg)

        # Publish global path using the class variable to accumulate poses
        self.global_path_msg.header.stamp = timestamp
        self.global_path_msg.header.frame_id = self.map_frame
        self.global_path_msg.poses.append(pose_msg)
        self.global_path_pub.publish(self.global_path_msg)

    def _publish_slam_path(self, timestamp):
        """Publish globally corrected SLAM path using thread-safe poses"""
        slam_path_msg = Path()
        slam_path_msg.header.stamp = timestamp
        slam_path_msg.header.frame_id = self.map_frame
        
        # Get poses in thread-safe manner
        with self._poses_lock:
            poses_copy = self._current_poses.copy() if self._current_poses else []
        
        for pose in poses_copy:
            p = PoseStamped()
            p.header = slam_path_msg.header
            p.pose.position.x = pose[0, 3]
            p.pose.position.y = pose[1, 3]
            p.pose.position.z = pose[2, 3]
            q = R.from_matrix(pose[:3, :3]).as_quat()
            p.pose.orientation.x = q[0]
            p.pose.orientation.y = q[1]
            p.pose.orientation.z = q[2]
            p.pose.orientation.w = q[3]
            slam_path_msg.poses.append(p)
        
        self.slam_path_pub.publish(slam_path_msg)

    def publish_occupancy_grid(self):
        """Publish 2D occupancy grid at 1Hz"""
        try:
            # Skip if no lidar data received yet
            if self.latest_lidar_timestamp is None:
                return
                
            # Compute occupancy information
            self.occupancy_mapper.compute_3d_occupancy_information()
            self.occupancy_mapper.compute_2d_occupancy_information()
            
            # Create OccupancyGrid message
            occupancy_msg = OccupancyGrid()
            occupancy_msg.header.stamp = self.latest_lidar_timestamp  # Use lidar timestamp
            occupancy_msg.header.frame_id = self.map_frame
            
            # Set map metadata
            occupancy_msg.info.resolution = self.occupancy_mapper.config.resolution
            occupancy_msg.info.width = int(self.occupancy_mapper.occupancy_grid.shape[0])
            occupancy_msg.info.height = int(self.occupancy_mapper.occupancy_grid.shape[1])
            
            occupancy_msg.info.origin.position.x = float(self.occupancy_mapper.lower_bound[0]) * self.occupancy_mapper.config.resolution
            occupancy_msg.info.origin.position.y = float(self.occupancy_mapper.lower_bound[1]) * self.occupancy_mapper.config.resolution
            occupancy_msg.info.origin.position.z = 0.0
            occupancy_msg.info.origin.orientation.x = 0.0
            occupancy_msg.info.origin.orientation.y = 0.0
            occupancy_msg.info.origin.orientation.z = 0.0
            occupancy_msg.info.origin.orientation.w = 1.0
            
            # Convert occupancy grid to ROS format
            # The occupancy grid values are in [0, 1] where 0 = free, 1 = occupied
            # ROS expects values in [0, 100] where 0 = free, 100 = occupied, -1 = unknown
            occupancy_data = self.occupancy_mapper.occupancy_grid.T.copy()
            occupancy_data = occupancy_data.flatten()
            
            # Vectorized conversion for ROS message
            free = occupancy_data < self.occupancy_mapper.config.free_threshold
            occupied = occupancy_data > self.occupancy_mapper.config.occupied_threshold
            ros_occupancy_data = np.full_like(occupancy_data, -1, dtype=np.int8)
            ros_occupancy_data[free] = 0
            ros_occupancy_data[occupied] = 100
            ros_occupancy_data = ros_occupancy_data.flatten().tolist()
            
            occupancy_msg.data = ros_occupancy_data
            
            # Publish the map
            self.map_pub.publish(occupancy_msg)
                
        except Exception as e:
            self.get_logger().warn(f'Failed to publish occupancy grid: {str(e)}')

    def destroy_node(self):
        """Clean shutdown with proper thread termination"""
        # Signal the SLAM thread to shutdown
        self.shutdown_event.set()
        
        # Wait for the SLAM thread to finish
        if self.slam_thread and self.slam_thread.is_alive():
            rclpy.logging.get_logger('kiss_slam_ros').info('Waiting for SLAM thread to finish...')
            self.slam_thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.slam_thread.is_alive():
                rclpy.logging.get_logger('kiss_slam_ros').warn('SLAM thread did not finish gracefully')
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt: shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
