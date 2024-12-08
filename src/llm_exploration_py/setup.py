from setuptools import find_packages, setup

package_name = 'llm_exploration_py'

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
            'find_unit_client = llm_exploration_py.llm_exploration_client:main',
            'find_unit_server = llm_exploration_py.llm_exploration_server:main',
            'get_unit_num_client = llm_exploration_py.get_unit_num_service_client:main',
            'get_unit_num_service = llm_exploration_py.get_unit_num_service_server:main'
        ],
    },
)
