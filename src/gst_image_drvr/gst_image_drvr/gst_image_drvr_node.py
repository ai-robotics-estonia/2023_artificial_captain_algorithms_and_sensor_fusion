import sys
import time
import queue
import signal
import threading

import rclpy
from rclpy.node import Node

from src.img_src import CameraSrc
from src.img_publisher import GstImgPublisher

VIDEO_FPS = 30

DEFAULT_COMPRESSION = 85

DEFAULT_FRAME_ID = "gst_image_drv"

# Text drawing defaults
TIME_DEFAULT_X = 10
TIME_DEFAULT_Y = 35
TIME_DEFAULT_SIZE = 1.2
TIME_DEFAULT_THICKNESS = 2

def declare_parameters(node: Node) -> None:
    """
    Declare parameters for the node.
    
    Args:
        node: ROS2 node object
    Returns:
        None
    """
    
    # Debugging
    node.declare_parameter('debug', False)
    
    # Timing
    node.declare_parameter('rate', 2.0)
    
    # Stability
    node.declare_parameter('no_frame_timeout_sec', 5.0)
    
    # GStreamer parameters
    node.declare_parameter('pipeline', '')
    
    # Camera frame
    node.declare_parameter('add_timestamp', False)
    node.declare_parameter('frame_id', DEFAULT_FRAME_ID)
    node.declare_parameter('camera_info', '')
    
    # QoS profile
    node.declare_parameter('buffer_depth', 1)
    node.declare_parameter('lease_duration', 1)
    node.declare_parameter('pub_unreliable', False)
    
    # Publishing
    node.declare_parameter('publish_image_raw', True)
    node.declare_parameter('publish_compressed', True)
    node.declare_parameter('publish_scaled_image_raw', False)
    node.declare_parameter('publish_scaled_compressed', False)
    node.declare_parameter('scaled_width', 0)
    node.declare_parameter('scaled_height', 0)
    node.declare_parameter('compression_quality', DEFAULT_COMPRESSION)
    
    # Timestamp position for full resolution image
    node.declare_parameter('time_x', TIME_DEFAULT_X)
    node.declare_parameter('time_y', TIME_DEFAULT_Y)
    node.declare_parameter('time_font_size', TIME_DEFAULT_SIZE)
    node.declare_parameter('time_font_thickness', TIME_DEFAULT_THICKNESS)
    
    # Timestamp position for scaled image
    node.declare_parameter('scaled_time_x', TIME_DEFAULT_X)
    node.declare_parameter('scaled_time_y', TIME_DEFAULT_Y)
    node.declare_parameter('scaled_time_font_size', TIME_DEFAULT_SIZE)
    node.declare_parameter('scaled_time_font_thickness', TIME_DEFAULT_THICKNESS)

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
    node = Node('OAK_D_Driver')
    
    # Declare parameters
    declare_parameters(node)
    
    # Mutex for thread-safe access to data queue
    mutex = threading.Lock()
    
    # Q for frames
    q = queue.Queue(maxsize=1)
    
    # Image event
    img_event = threading.Event()
    
    # Threads
    cam_drvr = CameraSrc(node, mutex, q, img_event)
    img_publisher = GstImgPublisher(node, mutex, q, img_event)
    
    # Start the threads
    cam_drvr.start()
    img_publisher.start()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
    
    finally:
        node.get_logger().info("Shutting down")
        
        # Stop the threads
        cam_drvr.stop()
        img_publisher.stop()
        
        # Wait for threads to finish
        cam_drvr.join()
        img_publisher.join()
        
        # Destroy the node
        node.destroy_node()
        
        # Shutdown ROS2
        rclpy.shutdown()
        
        # If you don't do this, ROS2 will complain 
        # that the process died with an error
        exit(0)

if __name__ == "__main__":
    main()
