import cv2
import time
import numpy as np

from threading import Thread

import rclpy
import rclpy.qos as qos
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from turbojpeg import TurboJPEG, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE
from src.common import ThreadInterface, Frame, create_sensor_qos_profile

from builtin_interfaces.msg import Time

class ImgSrc(Node):
    def __init__(self, thread_id, interface: ThreadInterface) -> None:
        """
        Class for the image source thread. It receives images from ROS2 topics and puts them in a queue.

        Args:
            id (int): The ID of the thread
            interface (ThreadInterface): The interface object that contains the queue, mutex, and event
            
        Returns:
            None
        """
        
        # Call the parent constructor
        super().__init__(f'ML_detector_subscriber_{thread_id}')
        
        self.id = thread_id
        
        self.logger = self.get_logger()
        
        self.logger.info(f"PROD_{self.id}: Initializing")
        
        # Debugging options
        self.declare_parameter('debug', 'False')
        
        # QoS settings
        self.declare_parameter('sub_buffer_depth', 1)
        self.declare_parameter('sub_lease_duration', 1)
        self.declare_parameter('sub_unreliable', False)

        # Inter-thread communication
        self.q = interface.get_queue()
        self.mutex = interface.get_lock()
        topic = interface.get_topic()
        
        # Read parameters
        
        # Debugging
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # QoS settings
        qos_sub_buffer_depth = self.get_parameter('sub_buffer_depth').get_parameter_value().integer_value
        qos_sub_lease_duration = self.get_parameter('sub_lease_duration').get_parameter_value().integer_value
        qos_sub_unreliable = self.get_parameter('sub_unreliable').get_parameter_value().bool_value
        
        # QoS profiles
        reliable_sensor_qos_profile = create_sensor_qos_profile(qos_sub_buffer_depth, qos_sub_lease_duration)
        pub_qos_profile = qos.qos_profile_sensor_data if qos_sub_unreliable else reliable_sensor_qos_profile
        
        # Time to sleep between iterations to maintain the desired rate
        self.last_publish_time = self.get_clock().now().to_msg()
        
        # OpenCV bridge for converting ROS images to numpy arrays
        self.bridge = CvBridge()
        
        topic = topic.replace('/detection', '/image_raw/compressed') # TODO: Remove this line when the topic names are fixed
        
        # Create image subscriber for the current camera
        sub_topic_type = CompressedImage if topic.endswith('/compressed') else Image
        sub_cb = self._compressed_img_callback if topic.endswith('/compressed') else self._img_callback
        
        self.sub = self.create_subscription(sub_topic_type, topic, sub_cb, qos_profile=pub_qos_profile)
        self.logger.info(f"PROD_{self.id}: Subscribing to {topic}")

        self.logger.info(f"PROD_{self.id}: Initialization complete")

    def _img_callback(self, msg: Image) -> None:
        """
        Callback function for the image subscriber. It receives the image from the subscriber,
        then converts it to a numpy array and puts it in the queue.
        
        Args:
            msg (Image): The message received from the subscriber
            
        Returns:
            None
        """
        
        if self.debug:
            self.logger.info(f"PROD_{self.id}: Received image at {msg.header.stamp.sec}")
        
        # Convert the image to a numpy array
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Get the timestamp
        timestamp = msg.header.stamp
        
        # Get the frame ID
        frame_id = msg.header.frame_id
        
        # Publish the image
        self._send_img(img, timestamp, frame_id)
        
    def _compressed_img_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the compressed image subscriber. It receives the image from the subscriber,
        decompresses it, then converts it to a numpy array and puts it in the queue.
        
        Args:
            msg (CompressedImage): The message received from the subscriber
            
        Returns:
            None
        """
        
        if self.debug:
            self.logger.info(f"PROD_{self.id}: Received compressed image at {msg.header.stamp.sec}")
        
        # Decompress the image
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Get the timestamp
        timestamp = msg.header.stamp
        
        # Get the frame ID
        frame_id = msg.header.frame_id
        
        # Publish the image
        self._send_img(img, timestamp, frame_id)
            
    def _send_img(self, img: np.ndarray, timestamp: Time, frame_id: str) -> None:
        """
        Send an image to the queue.

        Args:
            img (np.ndarray): The image to be sent
            timestamp (Time): The timestamp of the image
            frame_id (str): The frame ID of the image

        Returns:
            None.
        """
        
        # Create a Frame object
        frame = Frame(img, timestamp, frame_id)
        
        # Put the frame in the queue
        if not self.q.full():
            with self.mutex:
                self.q.put(frame)
        
            if self.debug:
                self.logger.info(f"PROD_{self.id}: Published image at {frame.get_timestmp_as_sec()}")