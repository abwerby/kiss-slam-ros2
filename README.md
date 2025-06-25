<div align="center">
    <h1>KISS-SLAM</h1>
    <a href="https://github.com/PRBonn/kiss-slam/releases"><img src="https://img.shields.io/github/v/release/PRBonn/kiss-slam?label=version" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PRBonn/kiss-slam" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>
    <br />
    <br />
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/README.md#Install">Install</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kiss2025iros.pdf">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://github.com/PRBonn/kiss-slam/issues>Contact Us</a>
  <br />
  <br />

[KISS-SLAM](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kiss2025iros.pdf) is a simple, robust, and accurate 3D LiDAR SLAM system that **just works**.


![motivation](https://github.com/user-attachments/assets/66c3e50f-009a-4a36-9856-283a895c300f)


</div>

<hr />

## Install

```
pip install kiss-slam
```

## Running the system
Next, follow the instructions on how to run the system by typing:
```
kiss_slam_pipeline --help
```

This should print the following help message:

![help](https://github.com/user-attachments/assets/5a6fe624-2aaf-466f-8a18-51039b794000)

### Config
You can generate a default `config.yaml` by typing:

```
kiss_slam_dump_config
```

which will generate a `kiss_slam.yaml` file. Now, you can modify the parameters and pass the file to the `--config` option when running the `kiss_slam_pipeline`.

## Running with ROS2

To run the ROS2 wrapper, you first need to build the ROS2 workspace.

#### Build the workspace

```bash
cd ros2
colcon build --symlink-install
```

#### Source the workspace

Before running the node, you need to source the workspace to make the packages available in your environment:

```bash
source ros2/install/setup.bash
```

#### Launch the SLAM node

Now you can launch the SLAM node using the provided launch file:

```bash
ros2 launch kiss_slam_ros slam.launch.py
```

This will start the `slam_node` which will subscribe to the `/points_raw` topic.

You can customize the launch by setting the following arguments:
*   `topic`: The input point cloud topic. Default: `/ouster/points`.
*   `visualize`: Whether to launch RViz. Default: `true`.
*   `bagfile`: Path to a rosbag to play. If provided.
*   `use_sim_time`: Whether to use simulation time. Default: `true`.
*   `base_frame`: The base frame of the robot. Default: `base_link`.
*   `odom_frame`: The odometry frame. Default: `odom`.

For example, to run with a different topic and without visualization and with s
```bash
ros2 launch kiss_slam_ros slam.launch.py bagfile:=/path/tobag topic:=/my_points visualize:=false
```


You can also change the SLAM parameters by editing the `ros2/src/kiss_slam_ros/config/params.yaml` file.


### Install Python API (developer mode)
For development purposes:

```
sudo apt install git python3-pip libeigen3-dev libsuitesparse-dev
python3 -m pip install --upgrade pip
git clone https://github.com/PRBonn/kiss-slam.git
cd kiss-slam
make editable
```

## Citation
If you use this library for any academic work, please cite our original paper:
```bib
@article{kiss2025arxiv,
  author   = {T. Guadagnino and B. Mersch and S. Gupta and I. Vizzo and G. Grisetti and C. Stachniss},
  title    = {{KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities}},
  journal  = {arXiv preprint},
  year     = 2025,
  volume   = {arXiv:2503.12660},
  url      = {https://arxiv.org/pdf/2503.12660},
}
```

## Acknowledgements
This project builds on top of [KISS-ICP](https://github.com/PRBonn/kiss-icp), [MapClosures](https://github.com/PRBonn/MapClosures), and [g2o](https://github.com/RainerKuemmerle/g2o).

