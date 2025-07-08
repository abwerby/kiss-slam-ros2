
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


    # Odometry node
    odometry_node = Node(
        package="kiss_slam_ros",
        executable="odometry_node",
        name="odometry_node",
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

    # Bag playback
    bagfile_play = ExecuteProcess(
        # --remap /j100_0819/tf:=/tf /j100_0819/tf_static:=/tf_static
        cmd=["ros2", "bag", "play", "--rate", "1", bagfile, "--clock", "1000.0", "--remap", "/j100_0819/tf:=/tf", "--remap", "/j100_0819/tf_static:=/tf_static"],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("topic", default_value="/j100_0000/sensors/lidar3d_0/points"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            odometry_node,
            bagfile_play,
        ]
    )
