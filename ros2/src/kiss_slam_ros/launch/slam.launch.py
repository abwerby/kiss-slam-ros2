from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    topic = LaunchConfiguration("topic", default="/ouster/points")
    visualize = LaunchConfiguration("visualize", default="true")
    bagfile = LaunchConfiguration("bagfile", default="")
    base_frame = LaunchConfiguration("base_frame", default="base_link")
    odom_frame = LaunchConfiguration("odom_frame", default="odom")
    map_frame = LaunchConfiguration("map_frame", default="map")

    # KISS-SLAM node
    slam_node = Node(
        package="kiss_slam_ros",
        executable="slam_node",
        name="slam_node",
        output="screen",
        remappings=[
            ("/points_raw", topic),
        ],
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("kiss_slam_ros"), "config", "params.yaml"]
            ),
            {
                "use_sim_time": use_sim_time,
                "base_frame": base_frame,
                "odom_frame": odom_frame,
                "map_frame": map_frame,
            }
        ],
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("kiss_slam_ros"), "rviz", "slam.rviz"]
            ),
        ],
        condition=IfCondition(visualize),
    )

    # Bag playback
    bagfile_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", "--rate", "2", bagfile, "--clock", "1000.0"],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("topic", default_value="/ouster/points"),
            DeclareLaunchArgument("visualize", default_value="true"),
            DeclareLaunchArgument("bagfile", default_value=""),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            slam_node,
            bagfile_play,
            rviz_node
        ]
    )