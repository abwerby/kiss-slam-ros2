import os
import time
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from map_closures import map_closures
from tqdm import tqdm, trange

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
from kiss_slam.slam import KissSLAM
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam_ros.utils.config import declare_parameters, get_kiss_slam_config


# setup rclpy logging
rclpy.logging.set_logger_level('kiss_slam_ros', rclpy.logging.LoggingSeverity.INFO)

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

        # Map saving parameters
        self.save_final_map = self.get_parameter('save_final_map').get_parameter_value().bool_value
        self.map_save_directory = self.get_parameter('map_save_directory').get_parameter_value().string_value
        self.save_2d_map = self.get_parameter('save_2d_map').get_parameter_value().bool_value
        self.save_3d_map = self.get_parameter('save_3d_map').get_parameter_value().bool_value

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
            f'ROS2 Parameters:\n'
            f'  Save Final Map: {self.save_final_map}\n'
            f'  Map Save Directory: {self.map_save_directory}\n'
            f'  Save 2D Map: {self.save_2d_map}\n'
            f'  Save 3D Map: {self.save_3d_map}\n'
        )

        self.slam = KissSLAM(config)
        self.occupancy_mapper = OccupancyGridMapper(config.occupancy_mapper)
        self.slam_config = config  # Store config for later use
        
        # Store scans for final map generation if map saving is enabled
        self.stored_scans: List[Tuple[np.ndarray, np.ndarray]] = []  # (points, timestamps)
        self.final_map_occupancy_mapper = None
        if self.save_final_map:
            self.final_map_occupancy_mapper = OccupancyGridMapper(config.occupancy_mapper)
        
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.slam_path_pub = self.create_publisher(Path, 'slam_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)
        self.local_pose_pub = self.create_publisher(PoseStamped, 'local_pose', 10)
        self.global_path_pub = self.create_publisher(Path, 'global_path', 10)
        self.global_pose_pub = self.create_publisher(PoseStamped, 'global_pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
    

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

        # Create timer for occupancy grid publishing at 1Hz
        self.map_timer = self.create_timer(.2, self.publish_occupancy_grid)
        
        # Store the latest lidar message timestamp for map publishing
        self.latest_lidar_timestamp = None

    def listener_callback(self, msg):
        """Fast odometry thread - processes scans and publishes odom->base_link transform"""
        # Extract points and timestamps from PointCloud2 message
        points_np = pc2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True)
        timestamps = points_np['t'].astype(np.float64)
        points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)

        # Odometry processing (no threading)
        self.slam.process_scan(points, timestamps)

        # Store scan for final map generation if enabled
        if self.save_final_map:
            self.stored_scans.append((points.copy(), timestamps.copy()))

        # Store the latest lidar timestamp for map publishing
        self.latest_lidar_timestamp = msg.header.stamp

        # get the last corrected local pose (odom->base_link)
        odom_T_base = self.slam.local_map_graph.last_local_map.local_trajectory[-1]

        # get the keypose of the last local map keypose (it should be relative to the first local map) (map->odom)
        map_T_odom = self.slam.get_keyposes()[-1] 
        
        # Integrate scan into occupancy mapper using aligned pose
        self.occupancy_mapper.integrate_frame(points, self.slam.poses[-1])

        # Create timestamp for this frame
        current_time = msg.header.stamp
        
        # Publish odom->base_link transform
        self._publish_odom_transform(odom_T_base, current_time)

        # Publish map->odom transform (using aligned transformation)
        self._publish_map_transform(map_T_odom, current_time)
        
        # Publish odometry pose and path
        self._publish_odometry_pose_and_path(odom_T_base, current_time)

        # Publish global pose from map to base_link (using aligned transformation)
        self._publish_global_path(map_T_odom, odom_T_base, current_time)
        
        # Publish corrected SLAM path
        self._publish_slam_path(current_time)


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
        """Publish globally corrected SLAM path"""
        slam_path_msg = Path()
        slam_path_msg.header.stamp = timestamp
        slam_path_msg.header.frame_id = self.map_frame
        
        for pose in self.slam.poses:
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

    def save_final_map_data(self):
        """Save the final optimized map using all stored scans and optimized poses"""
        if not self.save_final_map or not self.stored_scans:
            return
            
        try:
            print("Generating final optimized map...")
            
            # Get final optimized poses
            final_poses = self.slam.poses
            if len(final_poses) != len(self.stored_scans):
                print(f"Pose count ({len(final_poses)}) doesn't match scan count ({len(self.stored_scans)})")
                return
                
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join(self.map_save_directory, f"kiss_slam_map_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize final map occupancy mapper
            final_occupancy_mapper = OccupancyGridMapper(self.final_map_occupancy_mapper.config)
            
            # Get reference ground alignment from the first local map
            # if len(self.slam.local_map_graph.graph) > 0:
            #     first_local_map = self.slam.local_map_graph[0]
            #     ref_ground_alignment = map_closures.align_map_to_local_ground(
            #         first_local_map.pcd.point.positions.cpu().numpy(),
            #         self.slam_config.odometry.mapping.voxel_size,
            #     )
            # else:
            ref_ground_alignment = np.eye(4)
            
            print(f"Processing {len(self.stored_scans)} scans for final map generation...")
            
            # Process all stored scans with optimized poses
            for idx, ((points, timestamps), pose) in enumerate(zip(self.stored_scans, final_poses)):
                if idx % 100 == 0:  # Log progress every 100 scans
                    print(f"Processing scan {idx+1}/{len(self.stored_scans)}")
                    
                # Apply ground alignment and optimized pose
                aligned_pose = ref_ground_alignment @ pose
                final_occupancy_mapper.integrate_frame(points, aligned_pose)
            
            # Compute occupancy information
            print("Computing 3D occupancy information...")
            final_occupancy_mapper.compute_3d_occupancy_information()
            
            # Save 3D map if requested
            if self.save_3d_map:
                print("Saving 3D occupancy grid...")
                occupancy_3d_dir = os.path.join(output_dir, "occupancy_3d")
                os.makedirs(occupancy_3d_dir, exist_ok=True)
                final_occupancy_mapper.write_3d_occupancy_grid(occupancy_3d_dir)
            
            # Save 2D map if requested
            if self.save_2d_map:
                print("Computing and saving 2D occupancy grid...")
                final_occupancy_mapper.compute_2d_occupancy_information()
                occupancy_2d_dir = os.path.join(output_dir, "occupancy_2d")
                os.makedirs(occupancy_2d_dir, exist_ok=True)
                final_occupancy_mapper.write_2d_occupancy_grid(occupancy_2d_dir)
            
            # Save trajectory
            trajectory_file = os.path.join(output_dir, "trajectory.txt")
            with open(trajectory_file, 'w') as f:
                for pose in final_poses:
                    # Write pose as translation and quaternion
                    t = pose[:3, 3]
                    q = R.from_matrix(pose[:3, :3]).as_quat()  # [x, y, z, w]
                    f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            
            # Save pose graph
            if hasattr(self.slam, 'pose_graph'):
                graph_file = os.path.join(output_dir, "pose_graph.g2o")
                self.slam.pose_graph.write_graph(graph_file)
            
            print(f"Final map saved successfully to: {output_dir}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save final map: {str(e)}")

    def destroy_node(self):
        """Clean shutdown with optional map saving"""
        if self.save_final_map:
            # self.get_logger().info("Saving final optimized map before shutdown...")
            self.save_final_map_data()
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
