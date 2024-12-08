from setuptools import setup

package_name = 'llm_delivery'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wjh',
    maintainer_email='wjh_9696@163.com',
    description='LLM-based delivery.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_delivery_node = llm_delivery.llm_delivery_node:main',
            'robot_pose_pub_node = llm_delivery.robot_pose_pub_node:main',
            'osm_planner_node = llm_delivery.osm_planner_node:main'
        ],
    },
)
