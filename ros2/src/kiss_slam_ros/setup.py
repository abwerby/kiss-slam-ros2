from setuptools import setup
import os
from glob import glob

package_name = 'kiss_slam_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='abdelrhman.werby@ki.uni-stuttgart.de',
    description='ROS2 wrapper for kiss-slam',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = kiss_slam_ros.slam_node:main',
            'odometry_node = kiss_slam_ros.odometry_node:main'
        ],
    },
)
