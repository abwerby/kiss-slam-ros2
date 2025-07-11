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
    topic = LaunchConfiguration("topic", default="/j100_0000/sensors/lidar3d_0/points")
    visualize = LaunchConfiguration("visualize", default="true")
    bagfile = LaunchConfiguration("bagfile", default="")
    namespace = LaunchConfiguration("namespace", default="")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # SLAM node
    slam_node = Node(
        package="kiss_slam_ros",
        executable="slam_node",
        name="slam_node",
        namespace=namespace,
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("kiss_slam_ros"), "config", "slam_params.yaml"]
            ),
            {"use_sim_time": use_sim_time},
        ],
    )

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
            ),
            {"use_sim_time": use_sim_time}
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
                [FindPackageShare("kiss_slam_ros"), "rviz", "slam.rviz"]
            ),
        ],
        condition=IfCondition(visualize),
    )

    # Bag playback, use this yaml file to override QoS settings for the bag playback
    override_qos_yaml = PathJoinSubstitution([
        FindPackageShare('kiss_slam_ros'),
        'config',
        'override_qos.yaml'
    ])

    bagfile_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            '--rate', '5',
            bagfile,
            '--clock',
            # '--qos-profile-overrides-path', override_qos_yaml,
            '--remap', '/j100_0819/tf:=/tf',
            '--remap', '/j100_0819/tf_static:=/tf_static'
        ],
        output='screen',
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    # map saver 
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("topic", default_value="/ouster/points"),
            DeclareLaunchArgument("visualize", default_value="true"),
            DeclareLaunchArgument("bagfile", default_value=""),
            DeclareLaunchArgument("namespace", default_value=""),
            slam_node,
            odometry_node,
            bagfile_play,
            rviz_node,
        ]
    )