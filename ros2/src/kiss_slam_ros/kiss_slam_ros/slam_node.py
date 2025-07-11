import rclpy
from rclpy.logging import get_logger
import numpy as np
import message_filters
import time
import functools

from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import Subscriber, TimeSynchronizer

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
import std_msgs.msg
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from kiss_icp.voxelization import voxel_down_sample
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.loop_closer import LoopCloser
from kiss_slam.pose_graph_optimizer import PoseGraphOptimizer
from kiss_slam.voxel_map import VoxelMap

from kiss_slam_ros.utils.config import declare_parameters, get_kiss_slam_config


# TODO: this should be moved to a separate utility module
def timing_decorator(func):
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # If called from a ROS node, use the node's logger
        if args and hasattr(args[0], 'get_logger') and callable(args[0].get_logger):
            args[0].get_logger().info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        else:
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
        return result
    return wrapper

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # Parameters
        declare_parameters(self)
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.config = get_kiss_slam_config(self)
        
        # Core SLAM Backend
        self.closer = LoopCloser(self.config.loop_closer)
        local_map_config = self.config.local_mapper
        self.local_map_voxel_size = local_map_config.voxel_size
        self.voxel_grid = VoxelMap(self.local_map_voxel_size)
        self.odom_local_map = VoxelMap(self.local_map_voxel_size)
        self.local_map_graph = LocalMapGraph()
        self.local_map_splitting_distance = local_map_config.splitting_distance
        self.optimizer = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        self.closures = []

        # state variables
        self.local_maps = []  # List to store local maps
        self.voxel_maps = []  # List to store voxel maps
        self.traveled_distance = 0.0 # Distance traveled since the last local map split
        self.last_split_pose = np.eye(4)  # Last pose at which the local map was split

        # ROS Communications
        self.fast_callback_group = ReentrantCallbackGroup()
        self.slow_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._init_publishers()
        self._init_subscribers()

        self.get_logger().info(
            f'KISS-SLAM configuration:\n'
            f'  Map Frame: {self.map_frame}\n'
            f'  Odometry Frame: {self.odom_frame}\n'
            f'  Odometry Config: {self.config.odometry}\n'
            f'  Local Mapper Config: {self.config.local_mapper}\n'
            f'  Occupancy Mapper Config: {self.config.occupancy_mapper}\n'
            f'  Loop Closer Config: {self.config.loop_closer}\n'
            f'  Pose Graph Optimizer Config: {self.config.pose_graph_optimizer}\n'
        )
        
        self.get_logger().info("SLAM node initialized and waiting for keyframes.")

    def _init_publishers(self):
        """Initialize all ROS publishers."""
        qos = QoSProfile(durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        # self.slam_path_pub = self.create_publisher(Path, 'slam_path', path_qos)
        self.pose_pub = self.create_publisher(PoseStamped, 'global_pose', qos)
        self.global_voxel_map_pub = self.create_publisher(PointCloud2, 'global_voxel_map', qos)
        # self.create_timer(2.0, self.publish_2D_map, callback_group=self.slow_callback_group)  # Publish map every 2 seconds
        # self.create_timer(0.2, self.publish_slam_path, callback_group=self.slow_callback_group) # Only for debugging purposes and visualization

    def _init_subscribers(self):
        """Initialize message_filters subscribers to synchronize keyframe data."""
        # Subscribe to the local map and the corresponding odometry pose
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        deskewed_points_sub = Subscriber(
            self,
            PointCloud2,
            'deskewed_points',
            qos_profile=qos
        )
        odom_sub = Subscriber(
            self,
            Odometry,
            'odometry',
            qos_profile=qos
        )
        self.ts = TimeSynchronizer([deskewed_points_sub, odom_sub], 20)
        self.ts.registerCallback(self.keyframe_callback)

    def keyframe_callback(self, deskewed_points_msg: PointCloud2, odom_msg: Odometry):
        """
        This callback is triggered only when a synchronized pair of deskewed_points and odometry messages arrives.
        """
        stamp = odom_msg.header.stamp

        # 1. Extract data from messages
        points_np = pc2.read_points(deskewed_points_msg, field_names=("x", "y", "z"), skip_nans=True)
        keyframe_points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)
        current_keyframe_pose = self._msg_to_pose(odom_msg.pose.pose)
        relative_motion = np.linalg.inv(self.last_split_pose) @ current_keyframe_pose

        # 2. update voxel grid and local map graph
        mapping_frame = voxel_down_sample(keyframe_points, self.local_map_voxel_size)
        self.voxel_grid.integrate_frame(mapping_frame, relative_motion)
        self.odom_local_map.integrate_frame(keyframe_points, relative_motion)
        self.local_map_graph.last_local_map.local_trajectory.append(relative_motion)

        # 3. Update and publish the map->odom transform and SLAM path
        keyposes = self.get_keyposes()
        poses = self.poses
        self._publish_transform(keyposes[0], stamp, self.map_frame, self.odom_frame)            
        self._publish_pose(poses[-1], stamp, self.map_frame)

        # 4. update traveled distance and check if we need to split the local map
        current_step_distance = np.linalg.norm(relative_motion[:3, -1])
        
        # split local map if necessary
        if current_step_distance > self.local_map_splitting_distance:
            # update the last split pose for relative motion
            self.last_split_pose = np.copy(current_keyframe_pose)
            # append the current voxel grid to the list of voxel maps
            last_local_map = self.local_map_graph.last_local_map
            last_pose__last_local_map = last_local_map.local_trajectory[-1]
            self.voxel_maps.append(self.voxel_grid.open3d_pcd_with_normals())
            transformed_local_map = self._transform_points(self.odom_local_map.point_cloud(), np.linalg.inv(last_local_map.local_trajectory[-1]))
            self.odom_local_map.clear()
            query_id = last_local_map.id
            query_points = self.voxel_grid.point_cloud()
            self.local_map_graph.finalize_local_map(self.voxel_grid)
            self.voxel_grid.clear()
            self.voxel_grid.add_points(transformed_local_map)
            self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
            self.optimizer.add_factor(
                self.local_map_graph.last_id, query_id, last_pose__last_local_map, np.eye(6)
            )
            self._compute_closures(query_id, query_points)
            gmap = self._create_global_voxel_map()
            self.publish_pc2(gmap, frame_id=self.map_frame, stamp=stamp)

    def _msg_to_pose(self, pose_msg) -> np.ndarray:
        """Convert a nav_msgs/Odometry to a 4x4 numpy array."""
        p = pose_msg.position
        o = pose_msg.orientation
        pose = np.eye(4)
        pose[:3, 3] = [p.x, p.y, p.z]
        pose[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        return pose

    def _transform_points(self, pcd, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return pcd @ R.T + t

    def _publish_pose(self, pose: np.ndarray, stamp, frame_id: str):
        p = PoseStamped()
        p.header.stamp = stamp
        p.header.frame_id = frame_id
        p.pose.position.x = pose[0, 3]
        p.pose.position.y = pose[1, 3]
        p.pose.position.z = pose[2, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        self.pose_pub.publish(p)

    def _publish_transform(self, pose: np.ndarray, stamp, frame_id: str, child_frame_id: str):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, frame_id, child_frame_id
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pose[0, 3], pose[1, 3], pose[2, 3]
        q = R.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q[0], q[1], q[2], q[3]
        self.tf_broadcaster.sendTransform(t)

    def _poses_to_path_msg(self, poses, stamp=None, frame_id=None):
        """Convert the current poses to a Path message."""
        path = Path()
        path.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        path.header.frame_id = frame_id if frame_id is not None else self.map_frame
        for pose_np in poses:
            p = PoseStamped()
            p.header.stamp, p.header.frame_id = stamp, frame_id
            p.pose.position.x, p.pose.position.y, p.pose.position.z = pose_np[0, 3], pose_np[1, 3], pose_np[2, 3]
            q = R.from_matrix(pose_np[:3, :3]).as_quat()
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = q[0], q[1], q[2], q[3]
            path.poses.append(p)
        return path

    def publish_slam_path(self):
        """Publish the SLAM path as a Path message."""
        path_msg = self._poses_to_path_msg(self.poses, stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)
        self.slam_path_pub.publish(path_msg)

    def _transform_points(self, pcd, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return pcd @ R.T + t
    
    def _compute_closures(self, query_id, query):
        is_good, source_id, target_id, pose_constraint = self.closer.compute(
            query_id, query, self.local_map_graph
        )
        if is_good:
            self.closures.append((source_id, target_id))
            self.optimizer.add_factor(source_id, target_id, pose_constraint, np.eye(6))
            self._optimize_pose_graph()
    
    def _optimize_pose_graph(self):
        self.optimizer.optimize()
        estimates = self.optimizer.estimates()
        for id_, pose in estimates.items():
            self.local_map_graph[id_].keypose = np.copy(pose)

    def publish_pc2(self, points, frame_id=None, stamp=None):
        """Publish the points as a PointCloud2 message."""
        # Handle different input types
        if hasattr(points, 'point'):
            # Tensor-based PointCloud
            points_array = points.point.positions.cpu().numpy()
        elif hasattr(points, 'points'):
            # Legacy PointCloud
            points_array = np.asarray(points.points)
        else:
            # Numpy array
            points_array = points
        
        if points_array.shape[0] == 0:
            return
        
        # Create PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        header.frame_id = frame_id if frame_id is not None else self.map_frame
        global_voxel_map_msg = pc2.create_cloud_xyz32(header, points_array[:, :3])
        
        # Publish the global voxel map
        self.global_voxel_map_pub.publish(global_voxel_map_msg)
    
    def _create_global_voxel_map(self):
        """Create a global voxel map by transforming and combining all local voxel maps."""
        if not self.voxel_maps:
            return np.array([]).reshape(0, 3)
        
        poses = self.get_keyposes()
        # poses = self.poses  # Use the current poses from the local map graph
        all_points = []
        
        for i, voxel_map_pcd in enumerate(self.voxel_maps):
            if i >= len(poses):
                break
            
            # Convert tensor PointCloud to numpy array
            if hasattr(voxel_map_pcd, 'point'):
                # Tensor-based PointCloud
                points = voxel_map_pcd.point.positions.cpu().numpy()
            # check if it is a numpy array already
            elif isinstance(voxel_map_pcd, np.ndarray):
                points = voxel_map_pcd
            else:
                # Legacy PointCloud
                points = np.asarray(voxel_map_pcd.points)
            
            if points.shape[0] == 0:
                continue
            
            # Transform points to global frame
            pose = poses[i]
            R = pose[:3, :3]
            t = pose[:3, 3]
            global_points = points @ R.T + t
            all_points.append(global_points)
        
        if not all_points:
            return np.array([]).reshape(0, 3)
        
        # Combine all points
        combined_points = np.vstack(all_points)
        
        # Apply voxel downsampling for efficiency
        if combined_points.shape[0] > 10000:  # Only downsample if we have many points
            combined_points = voxel_down_sample(combined_points, self.local_map_voxel_size)
        
        return combined_points
        
    @timing_decorator
    def publish_2D_map(self):
        """Generate occupancy map using voxel maps and their corresponding key poses."""
        if not self.voxel_maps:
            return
        
        # Get the current key poses
        key_poses = self.get_keyposes()
        
        # Ensure we have matching number of voxel maps and key poses
        num_maps = min(len(self.voxel_maps), len(key_poses))
        if num_maps == 0:
            return
        
        # Create occupancy mapper
        occupancy_mapper = OccupancyGridMapper(self.config.occupancy_mapper)
        # Integrate all voxel maps with their corresponding key poses
        pcd = self._create_global_voxel_map()
        occupancy_mapper.integrate_frame(pcd, key_poses[0])
        # Compute occupancy information
        occupancy_mapper.compute_3d_occupancy_information()
        occupancy_mapper.compute_2d_occupancy_information()
        # Create and publish occupancy grid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.config.occupancy_mapper.resolution
        msg.info.width = occupancy_mapper.occupancy_grid.shape[0]
        msg.info.height = occupancy_mapper.occupancy_grid.shape[1]
        msg.info.origin.position.x = float(occupancy_mapper.lower_bound[0]) * self.config.occupancy_mapper.resolution
        msg.info.origin.position.y = float(occupancy_mapper.lower_bound[1]) * self.config.occupancy_mapper.resolution
        
        # Convert occupancy grid to ROS format
        grid = occupancy_mapper.occupancy_grid.T.flatten()
        free = grid < self.config.occupancy_mapper.free_threshold
        occupied = grid > self.config.occupancy_mapper.free_threshold
        grid[free] = 100      # Free cells
        grid[occupied] = 0  # Occupied cells
        grid[~(free | occupied)] = -1  # Unknown cells
        msg.data = (grid).astype(np.int8).tolist()
        
        self.map_pub.publish(msg)

    @property
    def poses(self):
        poses = [np.eye(4)]
        for node in self.local_map_graph.local_maps():
            for rel_pose in node.local_trajectory[1:]:
                poses.append(node.keypose @ rel_pose)
        return poses
    
    def get_keyposes(self):
        return list(self.local_map_graph.keyposes())
    
    def _fine_grained_optimization(self):
        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        id_ = 0
        pgo.add_variable(id_, self.local_map_graph[id_].keypose)
        pgo.fix_variable(id_)
        for node in self.local_map_graph.local_maps():
            odometry_factors = [
                np.linalg.inv(T0) @ T1
                for T0, T1 in zip(node.local_trajectory[:-1], node.local_trajectory[1:])
            ]
            for i, factor in enumerate(odometry_factors):
                pgo.add_variable(id_ + 1, node.keypose @ node.local_trajectory[i + 1])
                pgo.add_factor(id_ + 1, id_, factor, np.eye(6))
                id_ += 1
            pgo.fix_variable(id_ - 1)

        pgo.optimize()
        poses = [x for x in pgo.estimates().values()]
        return poses, pgo


    def destroy_node(self):
        self.get_logger().info("Shutting down SLAM node...")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()