from scipy.spatial.transform import Rotation as R
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2 
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs_py import point_cloud2 as pc2

from nav_msgs.msg import Path
import tf2_ros
import numpy as np
from kiss_slam.slam import KissSLAM
from kiss_slam_ros.utils.config import declare_parameters, get_kiss_slam_config

# setup rclpy logging
rclpy.logging.set_logger_level('kiss_slam_ros', rclpy.logging.LoggingSeverity.INFO)



def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


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
            f'  Pose Graph Optimizer Config: {config.pose_graph_optimizer}'
        )

        self.slam = KissSLAM(config)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.slam_path_pub = self.create_publisher(Path, 'slam_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)
        self.local_pose_pub = self.create_publisher(PoseStamped, 'local_pose', 10)
        self.global_path_pub = self.create_publisher(Path, 'global_path', 10)
        self.global_pose_pub = self.create_publisher(PoseStamped, 'global_pose', 10)
    

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

    def listener_callback(self, msg):
        """Fast odometry thread - processes scans and publishes odom->base_link transform"""
        # Extract points and timestamps from PointCloud2 message
        st_time = time.time()
        points_np = pc2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True)
        timestamps = points_np['t'].astype(np.float64)
        points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)

        # Odometry processing (no threading)
        self.slam.process_scan(points, timestamps)

        # get the last corrected local pose (odom->base_link)
        # current_odom_pose = self.slam.poses[-1]
        odom_T_base = self.slam.local_map_graph.last_local_map.local_trajectory[-1]

        # get the keypose of the last local map keypose (it should be realtive the first local map) (map->odom)
        map_T_odom = self.slam.get_keyposes()[-1] # the keuposes get corrected after optimization in case of loop closures


        # Publish local map as PointCloud2 every 5 seconds
        # local_map_in_global = transform_points(self.slam.voxel_grid.point_cloud(), current_node.keypose)
        # if not hasattr(self, 'last_local_map_pub_time'):
        #     self.last_local_map_pub_time = self.get_clock().now()
        #     self.local_map_pub = self.create_publisher(PointCloud2, 'local_map_in_global', 1)

        # now = self.get_clock().now()
        # if (now - self.last_local_map_pub_time).nanoseconds >= 5 * 1e9:
        #     local_map_points = local_map_in_global.astype(np.float32)
        #     header = msg.header
        #     header.frame_id = self.odom_frame
        #     local_map_pc2 = pc2.create_cloud_xyz32(header, local_map_points)
        #     self.local_map_pub.publish(local_map_pc2)
        #     self.last_local_map_pub_time = now


        # Create timestamp for this frame
        current_time = self.get_clock().now().to_msg()
        
        # Publish odom->base_link transform
        self._publish_odom_transform(odom_T_base, current_time)

        # Publish map->odom transform
        self._publish_map_transform(map_T_odom, current_time)
        
        # Publish odometry pose and path
        self._publish_odometry_pose_and_path(odom_T_base, current_time)

        # Publish global pose from map to base_link
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

    def destroy_node(self):
        """Clean shutdown"""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
