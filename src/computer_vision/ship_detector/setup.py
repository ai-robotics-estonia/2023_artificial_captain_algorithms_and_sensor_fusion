import os
from glob import glob
from setuptools import setup
from shutil import copytree

package_name = 'ship_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name + '/src', [package_name + '/src/bearing_finder.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/object_detection.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/img_src.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/img_publisher.py']),
        ('lib/' + package_name + '/src', [package_name + '/src/common.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/siamese_test.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/test_on_video.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/deepsort.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/__init__.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/siamese_train.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/__init__.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/generate_videos.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/show_results.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/detection.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/__init__.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/iou_matching.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/track.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/linear_assignment.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/kalman_filter.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/nn_matching.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort/tracker.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/application_util', [package_name + '/src/deepsort_tracker/deep_sort/application_util/__init__.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/application_util', [package_name + '/src/deepsort_tracker/deep_sort/application_util/preprocessing.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/application_util', [package_name + '/src/deepsort_tracker/deep_sort/application_util/image_viewer.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/application_util', [package_name + '/src/deepsort_tracker/deep_sort/application_util/visualization.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/evaluate_motchallenge.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/tools', [package_name + '/src/deepsort_tracker/deep_sort/tools/__init__.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/tools', [package_name + '/src/deepsort_tracker/deep_sort/tools/generate_detections.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort/tools', [package_name + '/src/deepsort_tracker/deep_sort/tools/freeze_model.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/deep_sort', [package_name + '/src/deepsort_tracker/deep_sort/deep_sort_app.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/det', [package_name + '/src/deepsort_tracker/det/det_ssd512.txt']),
        ('lib/' + package_name + '/src/deepsort_tracker/det', [package_name + '/src/deepsort_tracker/det/det_mask_rcnn.txt']),
        ('lib/' + package_name + '/src/deepsort_tracker/det', [package_name + '/src/deepsort_tracker/det/det_yolo3.txt']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/siamese_net.py']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/get_images.py']),
        ('lib/' + package_name + '/src/deepsort_tracker/ckpts', [package_name + '/src/deepsort_tracker/ckpts/model640_new_weights.pt']),
        ('lib/' + package_name + '/src/deepsort_tracker', [package_name + '/src/deepsort_tracker/siamese_dataloader.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='can',
    maintainer_email='canersu34@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detection = ship_detector.detection_node:main',
        ],
    },
)
