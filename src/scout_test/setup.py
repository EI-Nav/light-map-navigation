from setuptools import find_packages, setup

package_name = 'scout_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='wjh_9696@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "simple_nav_node = scout_test.simple_nav_node:main",
            "pub_mapodom_node = scout_test.pub_mapodom_node:main",
            "test_gazebo_node = scout_test.test_gazebo_node:main",
            "param_test_node = scout_test.param_test_node:main",
            "depth_generate_node = scout_test.depth_generate_node:main"

        ],
    },
)
