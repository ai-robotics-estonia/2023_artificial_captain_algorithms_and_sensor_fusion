from setuptools import setup

package_name = 'gst_image_drvr'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name + '/src', [package_name + '/src/img_src.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/img_publisher.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/common.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aju',
    maintainer_email='info@mindchip.ee',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gst_image_drvr_node = gst_image_drvr.gst_image_drvr_node:main'
        ],
    },
)
