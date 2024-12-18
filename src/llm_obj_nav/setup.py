from setuptools import find_packages, setup

package_name = 'llm_obj_nav'

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
    maintainer='lab417',
    maintainer_email='lab417@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_obj_nav_node = llm_obj_nav.llm_obj_nav_node:main',
            'multicamera_test_node = llm_obj_nav.multicamera_test_node:main',
            'GLEE_test_node = llm_obj_nav.GLEE_test_node:main',
            'gazebo_simulator_node = llm_obj_nav.gazebo_simulator_node:main'
        ],
    },
)
