#!/bin/bash

import rclpy
from rclpy.node import Node

from src.common import ThreadInterface
from src.img_src import ImgSrc
from src.img_publisher import ImgPublisher
from src.object_detection import ObjectDetection

from rclpy.executors import MultiThreadedExecutor

def declare_parameters(node: Node) -> None:
    """
    Declare parameters for the node.
    
    Args:
        node: ROS2 node object
    Returns:
        None
    """
    
    # Debugging options
    node.declare_parameter('debug', 'False')
    
    # Publishing options                  
    node.declare_parameter('rate', '2.0')     
    node.declare_parameter('publish_raw_img', 'False')
    node.declare_parameter('publish_compressed', 'False')
    node.declare_parameter('publish_scaled', 'False')
    node.declare_parameter('publish_scaled_compressed', 'False')
    
    node.declare_parameter('scaled_width', '1920')
    node.declare_parameter('scaled_height', '1080')
    node.declare_parameter('compression_quality', '75')
    
    node.declare_parameter('frame_id_prefix', 'detection_')
    
    # Stability
    node.declare_parameter('no_frame_timeout_sec', '5.0')

    # YOLO config
    node.declare_parameter('yolo_model')
    node.declare_parameter('input_size', '1280')
    node.declare_parameter('confidence_threshold', '0.5')
    node.declare_parameter('iou_threshold', '0.5')
    
    # Tracker config
    node.declare_parameter('enable_tracker', 'False')
    node.declare_parameter('tracker_type', 'bytetrack')
    
    # QoS settings
    node.declare_parameter('sub_buffer_depth', 1)
    node.declare_parameter('sub_lease_duration', 1)
    node.declare_parameter('sub_unreliable', False)
    
    node.declare_parameter('img_pub_buffer_depth', 1)
    node.declare_parameter('img_pub_lease_duration', 1)
    node.declare_parameter('img_pub_unreliable', False)
    
    node.declare_parameter('bbox_pub_buffer_depth', 1)
    node.declare_parameter('bbox_pub_lease_duration', 1)
    node.declare_parameter('bbox_pub_unreliable', False)
    
    # Output config
    node.declare_parameter('enable_bounding_boxes', 'True')
    node.declare_parameter('bbox_topic', 'bounding_boxes')
    
    # Camera config
    node.declare_parameter('num_cams', '1')


    # Camera config
    num_cams = node.get_parameter('num_cams').get_parameter_value().integer_value
    
    node.get_logger().info(f'Number of images: {num_cams}')
    
    # Declare camera topics
    for i in range(num_cams):
        node.get_logger().info(f'Declaring: cam_{i}_topic')
        node.declare_parameter(f'cam_{i}_topic', '')
        
    return num_cams

def get_camera_topics(node: Node, num_cams: int) -> list:
    """
    Get camera topics from parameters.
    
    Args:
        node: ROS2 node object
        num_cams: Number of cameras
        
    Returns:
        cam_topics: List of camera topics
    """    
    
    cam_topics = []
    
    for i in range(num_cams):
        cam_topics.append(node.get_parameter(f'cam_{i}_topic').get_parameter_value().string_value)
    
    for i, topic in enumerate(cam_topics):
        node.get_logger().info(f'Camera topic: {topic}')
        if topic == '':
            node.get_logger().error(f'Camera topic {i} is empty! Please check the launch file.')
            raise ValueError(f'Camera topic {i} is empty! Please check the launch file.')
        
    return cam_topics

def create_thread_interfaces(node: Node, cam_topics: list, publish_img: bool) -> None:
    """
    Create thread interfaces for the node.
    
    Args:
        node: ROS2 node object
        cam_topics: List of camera topics
        publish_img: Whether to publish images or not
        
    Returns:
        sub_ifs: List of subscriber thread interfaces
        pub_ifs: List of publisher thread interfaces
    """
    
    sub_ifs = []
    pub_ifs = []
    
    for i, topic in enumerate(cam_topics):
        sub_if = ThreadInterface(i, topic)
        sub_ifs.append(sub_if)
        
        if publish_img:
            pub_topic = topic.replace('/compressed', '')
            pub_topic = pub_topic.replace('/image_raw', '')
            pub_topic += '/detection'
            
            pub_if = ThreadInterface(i, pub_topic)
            pub_ifs.append(pub_if)
            
    return sub_ifs, pub_ifs

def main(args=None) -> None:
    """
    Main function for the node.
    
    Args:
        args: Command line arguments
    Returns:
        None
    """
    
    rclpy.init(args=args)
    
    # Create the node
    node = Node('ML_detector')
    
    # Declare parameters
    num_cams = declare_parameters(node)
    
    # Get camera topics
    cam_topics = get_camera_topics(node, num_cams)
    
    # Get if we need to publish images
    raw_img_publish = node.get_parameter('publish_raw_img').get_parameter_value().bool_value
    compressed_img_publish = node.get_parameter('publish_compressed').get_parameter_value().bool_value
    publish_scaled = node.get_parameter('publish_scaled').get_parameter_value().bool_value
    publish_scaled_compressed = node.get_parameter('publish_scaled_compressed').get_parameter_value().bool_value
    
    publish_img = raw_img_publish or compressed_img_publish or publish_scaled or publish_scaled_compressed
    
    sub_ifs, pub_ifs = create_thread_interfaces(node, cam_topics, publish_img)
    
    sub_threads = []
    pub_threads = []    
    
    # Create publiser threads
    if publish_img:
        for i, _ in enumerate(cam_topics):
            pub_thread = ImgPublisher(i, pub_ifs[i])
            pub_threads.append(pub_thread)
    
    # Create subscriber threads
    for i, _ in enumerate(cam_topics):
        sub_thread = ImgSrc(i, sub_ifs[i])
        sub_threads.append(sub_thread)
    
    # Create object detection thread
    obj_det = ObjectDetection(sub_ifs, pub_ifs, publish_img)
    
    # Needed for parallel execution
    executor = MultiThreadedExecutor(num_threads=6)
    
    # Add nodes to executor
    executor.add_node(node)
    
    # Publisher threads
    for thread in pub_threads:
        executor.add_node(thread)
    
    # Object detection thread
    executor.add_node(obj_det)
    
    # Subscriber threads
    for thread in sub_threads:
        executor.add_node(thread)
    
    # Spin the executor
    try:
        executor.spin()
    
    finally:
        node.get_logger().info("Shutting down")
        
        rclpy.shutdown()
        
        # If you don't do this, ROS2 will complain 
        # that the process died with an error
        exit(0)
    
if __name__ == "__main__":
    main()
