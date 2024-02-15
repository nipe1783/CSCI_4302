from setuptools import setup, find_packages

package_name = 'nipe1783_hw2'

setup(
    name=package_name,
    version='0.0.0',
    # Assuming your package's Python code might be in a subdirectory, adjust accordingly
    packages=find_packages(),  # Automatically discover all packages and subpackages
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nic',
    maintainer_email='nipe1783@colorado.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'client = {package_name}.string_reversal_client:main',
            f'service = {package_name}.string_reversal_service:main',
            f'publisher = {package_name}.string_publisher:main',
            f'subscriber = {package_name}.string_subscriber:main',
        ],
    },
)
