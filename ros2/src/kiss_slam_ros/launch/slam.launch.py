from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.actions import Shutdown
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    topic = LaunchConfiguration("topic", default="/ouster/points")
    visualize = LaunchConfiguration("visualize", default="true")
    bagfile = LaunchConfiguration("bagfile", default="")

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
            )
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
            "require=true",
        ],
        condition=IfCondition(visualize),
    )

    # Bag playback
    bagfile_play = ExecuteProcess(
        # --remap /j100_0819/tf:=/tf /j100_0819/tf_static:=/tf_static
        cmd=["ros2", "bag", "play", "--rate", "2", bagfile, "--clock", "1000.0", "--remap", "/j100_0819/tf:=/tf", "--remap", "/j100_0819/tf_static:=/tf_static"],
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
            DeclareLaunchArgument("save_final_map", default_value="true"),
            DeclareLaunchArgument("map_save_directory", default_value="maps/map"),
            DeclareLaunchArgument("map_save_interval",default_value="10.0"),
            slam_node,
            bagfile_play,
            rviz_node,
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-c",
                    [
                        "while true; do mkdir -p $(dirname ",
                        LaunchConfiguration("map_save_directory"),
                        "); ros2 run nav2_map_server map_saver_cli -f ",
                        LaunchConfiguration("map_save_directory"),
                        "; sleep ",
                        LaunchConfiguration("map_save_interval"),
                        "; done",
                    ],
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("save_final_map")),
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bagfile_play,
                    on_exit=[Shutdown()]
                )
            ),
        ]
    )