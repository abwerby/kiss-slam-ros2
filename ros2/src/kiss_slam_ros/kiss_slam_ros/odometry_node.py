import rclpy
import numpy as np
import time
import threading
import queue
import std_msgs.msg

from kiss_icp.kiss_icp import KissICP
from kiss_icp.voxelization import voxel_down_sample

from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Path

import tf2_ros
from kiss_slam_ros.utils.config import declare_parameters, get_kiss_icp_config


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Declare and get parameters
        declare_parameters(self)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        config = get_kiss_icp_config(self)
        self.get_logger().info("KISS-ICP ROS2 node configured successfully.")

        # Core Odometry component
        self.odometry = KissICP(config)
        
        self.odometry_poses = []
        self.local_map_splitting_distance = self.get_parameter('kiss_slam.local_mapper.splitting_distance').value if self.has_parameter('kiss_slam.local_mapper.splitting_distance') else 5.0
        self.last_local_map = None
        self.last_deskewed_frame = None

        # ROS Communications
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._init_publishers()
        self._init_subscribers_and_timers()
        
        self.get_logger().info(
            f"Odometry Node initialized with odom_frame: {self.odom_frame}, base_frame: {self.base_frame}"
            f" and local_map_splitting_distance: {self.local_map_splitting_distance}"
        )
        self.get_logger().info("Odometry Node initialized. Waiting for data...")

    def _init_publishers(self):
        """Initialize all ROS publishers."""
        path_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odom_path_pub = self.create_publisher(Path, '/odom_path', path_qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/odom_pose', path_qos)
        self.deskewed_pub = self.create_publisher(PointCloud2, '/deskewed_points', 10)
        self.local_map_pub = self.create_publisher(PointCloud2, '/local_map', path_qos)

    def _init_subscribers_and_timers(self):
        """Initialize subscribers and timers with dedicated callback groups."""
        self.fast_callback_group = ReentrantCallbackGroup()
        self.slow_callback_group = MutuallyExclusiveCallbackGroup()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(
            PointCloud2, '/points_raw', self.listener_callback, qos_profile, callback_group=self.fast_callback_group
        )
        
        self.create_timer(0.5, self.publishing_callback, callback_group=self.slow_callback_group)

    def listener_callback(self, msg: PointCloud2):
        """
        FAST, REAL-TIME THREAD.
        Only performs odometry and publishes odom->base_link transform.
        """
        msg_fields = {field.name: field for field in msg.fields}
        if 't' in msg_fields:
            points_np = pc2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True)
            timestamps = points_np['t'].astype(np.float64)
        else:
            points_np = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            timestamps = np.array([])
            
        points = np.vstack([points_np['x'], points_np['y'], points_np['z']]).T.astype(np.float64)

        # Run KISS-ICP odometry
        deskewed_frame, _ = self.odometry.register_frame(points, timestamps)
        current_odom_pose = self.odometry.last_pose
        self.odometry_poses.append(current_odom_pose.copy())

        # TODO: remove the magic number 3.0, make it a parameter
        local_map = self.odometry.local_map.point_cloud()
        if np.linalg.norm(current_odom_pose[:3, -1]) > self.local_map_splitting_distance:
            rclpy.logging.get_logger("OdometryNode").info(
                f"New local map generated at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
            )
            transformed_local_map = self._transform_points(local_map, np.linalg.inv(current_odom_pose))
            self.odometry.local_map.clear()
            self.odometry.local_map.add_points(transformed_local_map)
            self.odometry.last_pose = np.eye(4)

        self.last_local_map = local_map
        self.last_deskewed_frame = deskewed_frame
        # Publish the odom -> base_link transform IMMEDIATELY
        self._publish_transform(current_odom_pose, msg.header.stamp, self.odom_frame, self.base_frame)
        self._publish_pose(current_odom_pose, msg.header.stamp, self.odom_frame)
        self._publish_deskewed_points(deskewed_frame, msg.header.stamp)
        self._publish_local_map(local_map, msg.header.stamp)


    def publishing_callback(self):
        """
        SLOW THREAD. Publishes paths at a fixed low rate.
        """
        if not self.odometry_poses:
            return

        stamp = self.get_clock().now().to_msg()
        
        # Publish odometry path
        odom_path_msg = self._poses_to_path_msg(self.odometry_poses, stamp, self.odom_frame)
        self.odom_path_pub.publish(odom_path_msg)

    
    def _publish_local_map(self, local_map: np.ndarray, stamp):
        """Publish the local map as a PointCloud2 message."""
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = self.odom_frame

        # Convert local_map to PointCloud2 message
        local_map_cloud = pc2.create_cloud_xyz32(header, local_map[:, :3])

        # Publish the local map
        self.local_map_pub.publish(local_map_cloud)

    def _publish_transform(self, pose: np.ndarray, stamp, frame_id: str, child_frame_id: str):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = pose[0, 3]
        t.transform.translation.y = pose[1, 3]
        t.transform.translation.z = pose[2, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

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

    def _publish_deskewed_points(self, deskewed_frame: np.ndarray, stamp):
        """Publish the deskewed point cloud."""
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = self.base_frame

        # Convert deskewed_frame to PointCloud2 message
        deskewed_cloud = pc2.create_cloud_xyz32(header, deskewed_frame[:, :3])

        # Publish the deskewed point cloud
        self.deskewed_pub.publish(deskewed_cloud)

    def _poses_to_path_msg(self, poses: list, stamp, frame_id: str) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame_id
        for pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = frame_id
            pose_stamped.pose.position.x = pose[0, 3]
            pose_stamped.pose.position.y = pose[1, 3]
            pose_stamped.pose.position.z = pose[2, 3]
            quat = R.from_matrix(pose[:3, :3]).as_quat()
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            path.poses.append(pose_stamped)
        return path

    def _transform_points(self, pcd, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return pcd @ R.T + t

    def destroy_node(self):
        self.get_logger().info("Shutting down odometry node.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    # executor = MultiThreadedExecutor()
    # executor.add_node(node)
    try:
        # executor.spin()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
