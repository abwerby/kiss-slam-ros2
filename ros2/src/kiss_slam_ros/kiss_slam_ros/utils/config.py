from rclpy.node import Node
from kiss_slam.config.config import (
    KissSLAMConfig, 
    KissOdometryConfig,
    LocalMapperConfig,
    OccupancyMapperConfig,
    LoopCloserConfig,
    PoseGraphOptimizerConfig
)
from kiss_icp.config.parser import KISSConfig
from kiss_icp.config.config import (
    AdaptiveThresholdConfig,
    DataConfig,
    MappingConfig,
    RegistrationConfig,
)
from map_closures.config.config import MapClosuresConfig

def get_param_value(node: Node, param_name, default_value=None):
    try:
        param = node.get_parameter(param_name)
        if param.type_ == param.Type.NOT_SET:
            return default_value
        elif param.type_ == param.Type.DOUBLE:
            value = param.get_parameter_value().double_value
            return None if value == -1.0 else value
        elif param.type_ == param.Type.INTEGER:
            value = param.get_parameter_value().integer_value
            return None if value == -1 else value
        elif param.type_ == param.Type.BOOL:
            return param.get_parameter_value().bool_value
        elif param.type_ == param.Type.STRING:
            value = param.get_parameter_value().string_value
            return None if value.lower() == 'none' or value == '' else value
        else:
            return default_value
    except Exception:
        return default_value

def declare_parameters(node: Node):
    node.declare_parameter('map_frame', 'map')
    node.declare_parameter('odom_frame', 'odom')
    node.declare_parameter('base_frame', 'base_link')
    node.declare_parameter('loop_closure_threshold', 5)
    node.declare_parameter('optimization_frequency', 1.0)
    
    # KISS-SLAM odometry parameters
    node.declare_parameter('kiss_slam.odometry.preprocessing.max_range', 100.0)
    node.declare_parameter('kiss_slam.odometry.preprocessing.min_range', 0.0)
    node.declare_parameter('kiss_slam.odometry.preprocessing.deskew', True)
    node.declare_parameter('kiss_slam.odometry.registration.max_num_iterations', 500)
    node.declare_parameter('kiss_slam.odometry.registration.convergence_criterion', 0.0001)
    node.declare_parameter('kiss_slam.odometry.registration.max_num_threads', 0)
    node.declare_parameter('kiss_slam.odometry.mapping.voxel_size', -1.0)
    node.declare_parameter('kiss_slam.odometry.mapping.max_points_per_voxel', 20)
    node.declare_parameter('kiss_slam.odometry.adaptive_threshold.fixed_threshold', -1.0)
    node.declare_parameter('kiss_slam.odometry.adaptive_threshold.initial_threshold', 2.0)
    node.declare_parameter('kiss_slam.odometry.adaptive_threshold.min_motion_th', 0.1)
    
    # Local mapper parameters
    node.declare_parameter('kiss_slam.local_mapper.voxel_size', 0.5)
    node.declare_parameter('kiss_slam.local_mapper.splitting_distance', 100.0)
    
    # Occupancy mapper parameters
    node.declare_parameter('kiss_slam.occupancy_mapper.free_threshold', 0.2)
    node.declare_parameter('kiss_slam.occupancy_mapper.occupied_threshold', 0.65)
    node.declare_parameter('kiss_slam.occupancy_mapper.resolution', 0.5)
    node.declare_parameter('kiss_slam.occupancy_mapper.max_range', -1.0)
    node.declare_parameter('kiss_slam.occupancy_mapper.z_min', 0.1)
    node.declare_parameter('kiss_slam.occupancy_mapper.z_max', 0.5)
    
    # Loop closer parameters
    node.declare_parameter('kiss_slam.loop_closer.detector.density_map_resolution', 0.5)
    node.declare_parameter('kiss_slam.loop_closer.detector.density_threshold', 0.05)
    node.declare_parameter('kiss_slam.loop_closer.detector.hamming_distance_threshold', 50)
    node.declare_parameter('kiss_slam.loop_closer.detector.inliers_threshold', 5)
    node.declare_parameter('kiss_slam.loop_closer.overlap_threshold', 0.4)
    
    # Pose graph optimizer parameters
    node.declare_parameter('kiss_slam.pose_graph_optimizer.max_iterations', 10)
    
    # Map saving parameters
    node.declare_parameter('save_final_map', False)
    node.declare_parameter('map_save_directory', '/tmp/kiss_slam_maps')
    node.declare_parameter('save_2d_map', True)
    node.declare_parameter('save_3d_map', True)

def get_kiss_icp_config(node: Node) -> KISSConfig:
    """Get KISS-ICP configuration for odometry node."""
    # Create the KissOdometryConfig first to match the SLAM structure
    odometry_config = KissOdometryConfig(
        preprocessing=DataConfig(
            max_range=get_param_value(node, 'kiss_slam.odometry.preprocessing.max_range', 100.0),
            min_range=get_param_value(node, 'kiss_slam.odometry.preprocessing.min_range', 0.0),
            deskew=get_param_value(node, 'kiss_slam.odometry.preprocessing.deskew', True),
        ),
        registration=RegistrationConfig(
            max_num_iterations=get_param_value(node, 'kiss_slam.odometry.registration.max_num_iterations', 500),
            convergence_criterion=get_param_value(node, 'kiss_slam.odometry.registration.convergence_criterion', 0.0001),
            max_num_threads=get_param_value(node, 'kiss_slam.odometry.registration.max_num_threads', 0),
        ),
        mapping=MappingConfig(
            voxel_size=get_param_value(node, 'kiss_slam.odometry.mapping.voxel_size', None),
            max_points_per_voxel=get_param_value(node, 'kiss_slam.odometry.mapping.max_points_per_voxel', 20),
        ),
        adaptive_threshold=AdaptiveThresholdConfig(
            fixed_threshold=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.fixed_threshold', None),
            initial_threshold=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.initial_threshold', 2.0),
            min_motion_th=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.min_motion_th', 0.1),
        )
    )

    if odometry_config.mapping.voxel_size is None:
        odometry_config.mapping.voxel_size = float(odometry_config.preprocessing.max_range / 100.0)
    
    # Convert to KISSConfig using the kiss_icp_config() method
    config = KISSConfig(
        data=odometry_config.preprocessing,
        registration=odometry_config.registration,
        mapping=odometry_config.mapping,
        adaptive_threshold=odometry_config.adaptive_threshold,
    )
        
    return config

def get_kiss_slam_config(node: Node) -> KissSLAMConfig:
    """Get full KISS-SLAM configuration for SLAM node."""
    # Use default odometry config since SLAM node doesn't need detailed odometry params
    default_odometry_config = KissOdometryConfig(
        preprocessing=DataConfig(
            max_range=get_param_value(node, 'kiss_slam.odometry.preprocessing.max_range', 100.0),
            min_range=get_param_value(node, 'kiss_slam.odometry.preprocessing.min_range', 0.0),
            deskew=get_param_value(node, 'kiss_slam.odometry.preprocessing.deskew', True),
        ),
        registration=RegistrationConfig(
            max_num_iterations=get_param_value(node, 'kiss_slam.odometry.registration.max_num_iterations', 500),
            convergence_criterion=get_param_value(node, 'kiss_slam.odometry.registration.convergence_criterion', 0.0001),
            max_num_threads=get_param_value(node, 'kiss_slam.odometry.registration.max_num_threads', 0),
        ),
        mapping=MappingConfig(
            voxel_size=get_param_value(node, 'kiss_slam.odometry.mapping.voxel_size', None),
            max_points_per_voxel=get_param_value(node, 'kiss_slam.odometry.mapping.max_points_per_voxel', 20),
        ),
        adaptive_threshold=AdaptiveThresholdConfig(
            fixed_threshold=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.fixed_threshold', None),
            initial_threshold=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.initial_threshold', 2.0),
            min_motion_th=get_param_value(node, 'kiss_slam.odometry.adaptive_threshold.min_motion_th', 0.1),
        )
    )
    
    # Apply voxel size defaults
    if default_odometry_config.mapping.voxel_size is None:
        default_odometry_config.mapping.voxel_size = float(default_odometry_config.preprocessing.max_range / 100.0)
    
    config = KissSLAMConfig(
        odometry=default_odometry_config,
        local_mapper=LocalMapperConfig(
            voxel_size=get_param_value(node, 'kiss_slam.local_mapper.voxel_size', 0.5),
            splitting_distance=get_param_value(node, 'kiss_slam.local_mapper.splitting_distance', 100.0),
        ),
        occupancy_mapper=OccupancyMapperConfig(
            free_threshold=get_param_value(node, 'kiss_slam.occupancy_mapper.free_threshold', 0.2),
            occupied_threshold=get_param_value(node, 'kiss_slam.occupancy_mapper.occupied_threshold', 0.65),
            resolution=get_param_value(node, 'kiss_slam.occupancy_mapper.resolution', 0.5),
            max_range=get_param_value(node, 'kiss_slam.occupancy_mapper.max_range', None),
            z_min=get_param_value(node, 'kiss_slam.occupancy_mapper.z_min', 0.1),
            z_max=get_param_value(node, 'kiss_slam.occupancy_mapper.z_max', 0.5),
        ),
        loop_closer=LoopCloserConfig(
            detector=MapClosuresConfig(
                density_map_resolution=get_param_value(node, 'kiss_slam.loop_closer.detector.density_map_resolution', 0.5),
                density_threshold=get_param_value(node, 'kiss_slam.loop_closer.detector.density_threshold', 0.05),
                hamming_distance_threshold=get_param_value(node, 'kiss_slam.loop_closer.detector.hamming_distance_threshold', 50),
                inliers_threshold=get_param_value(node, 'kiss_slam.loop_closer.detector.inliers_threshold', 5),
            ),
            overlap_threshold=get_param_value(node, 'kiss_slam.loop_closer.overlap_threshold', 0.4),
        ),
        pose_graph_optimizer=PoseGraphOptimizerConfig(
            max_iterations=get_param_value(node, 'kiss_slam.pose_graph_optimizer.max_iterations', 10),
        )
    )

    if config.occupancy_mapper.max_range is None:
        config.occupancy_mapper.max_range = default_odometry_config.preprocessing.max_range
        
    return config
