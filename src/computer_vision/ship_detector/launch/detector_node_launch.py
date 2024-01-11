import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    # args that can be set from the command line or a default will be used
    input_size_launch_arg = DeclareLaunchArgument(
        "input_size", default_value=TextSubstitution(text="640")
    )
    confidence_threshold_launch_arg = DeclareLaunchArgument(
        "confidence_threshold", default_value=TextSubstitution(text="0.3")
    )
    iou_threshold_launch_arg = DeclareLaunchArgument(
        "iou_threshold", default_value=TextSubstitution(text="0.3")
    )
    tracker_launch_arg = DeclareLaunchArgument(
        "tracker", default_value=TextSubstitution(text="bytetrack.yaml")
    )
    yolo_model_launch_arg = DeclareLaunchArgument(
        "yolo_model", default_value=TextSubstitution(text="/media/agx/yolo_weights/rt_models/fp16/640/yolov8n.engine")
    )
    detector_node_with_parameters = Node(
            package='ship_detector',
            executable='yolo_detection',
            name='detector',
            parameters=[{
                "input_size": LaunchConfiguration('input_size'),
                "confidence_threshold": LaunchConfiguration('confidence_threshold'),
                "iou_threshold": LaunchConfiguration('iou_threshold'),
                "tracker": LaunchConfiguration('tracker'),
                "yolo_model": LaunchConfiguration('yolo_model'),}
            ])



    return LaunchDescription([
        input_size_launch_arg,
        confidence_threshold_launch_arg,
        iou_threshold_launch_arg,
        tracker_launch_arg,
        yolo_model_launch_arg,
        detector_node_with_parameters
    ])
