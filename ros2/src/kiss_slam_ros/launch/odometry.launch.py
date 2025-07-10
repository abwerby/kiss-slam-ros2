
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

def generate_launch_description():
    # Launch arguments
    topic = LaunchConfiguration("topic", default="/ouster/points")
    bagfile = LaunchConfiguration("bagfile", default="")
    namespace = LaunchConfiguration("namespace", default="")
    visualize = LaunchConfiguration("visualize", default="true")


    # Odometry node
    odometry_node = Node(
        package="kiss_slam_ros",
        executable="odometry_node",
        name="odometry_node",
        namespace=namespace,
        output="screen",
        remappings=[
            ("/points_raw", topic),
        ],
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("kiss_slam_ros"), "config", "odometry_params.yaml"]
            )
        ],
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        namespace=namespace,
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("kiss_slam_ros"), "rviz", "odom.rviz"]
            ),
        ],
        condition=IfCondition(visualize),
    )

    # Bag playback
    bagfile_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            bagfile,
            "--rate", "2",
            "--clock",
            "--remap", "/j100_0819/tf:=/tf",
            "--remap", "/j100_0819/tf_static:=/tf_static",
            "--remap", "/ouster/tf_static:=/tf_static",
        ],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("visualize", default_value="true"),
            DeclareLaunchArgument("topic", default_value="/ouster/points"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("namespace", default_value=""),
            odometry_node,
            bagfile_play,
            rviz_node,
        ]
    )
