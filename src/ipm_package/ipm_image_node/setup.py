from setuptools import setup
import os
from glob import glob

package_name = 'ipm_image_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='florian',
    maintainer_email='git@flova.de',
    description='Inverse Perspective Mapping Node for Image or Mask Topics',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ipm = ipm_image_node.ipm:main',
            'ipm_test_onepoint = ipm_image_node.ipm_test_onepoint:main',
            'ipm_test_numpoints = ipm_image_node.ipm_test_numpoints:main',
            'ipm_json = ipm_image_node.ipm_json:main',
            'ipm_json2 = ipm_image_node.ipm_json2:main',
            'ipm_obstacle = ipm_image_node.ipm_obstacle:main',
            'ipm_obs_use = ipm_image_node.ipm_obs_use:main',
            'ipm_obstacle_server = ipm_image_node.ipm_obstacle_server:main',
            'ipm_obstacle_client = ipm_image_node.ipm_obstacle_client:main'
        ],
    },
)
