import cv2
import time
import datetime
import numpy as np

import rclpy
import rclpy.qos as qos
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from src.common import ThreadInterface, create_sensor_qos_profile

class MLErrorCodes:
    NO_ERROR = 0
    ML_ERROR = 1

TIME_DEFAULT_X = 10
TIME_DEFAULT_Y = 35
TIME_DEFAULT_SIZE = 1.2
TIME_DEFAULT_THICKNESS = 2
SCALED_TIME_DEFAULT_SIZE = 0.45
SCALED_TIME_DEFAULT_THICKNESS = 1

def cv_outline_text(img: np.ndarray, text: str, coords: tuple, font_size: float, thickness: int = 1,
                    fg_color: tuple = (255, 255, 255), bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Add outlined text to an OpenCV image.

    This function adds the specified text to an OpenCV image with an outline
    around the text. The function uses the specified font size and thickness
    for the text and outline. The foreground and background colors of the text
    and outline can also be specified.

    Args:
        img (numpy.ndarray): The OpenCV image to add text to.
        text (str): The text to add to the image.
        coords (tuple): The (x, y) coordinates of the text's bottom-left corner.
        font_size (float): The font size to use for the text.
        thickness (int, optional): The thickness of the text and outline (default is 1).
        fg_color (tuple, optional): The color of the text (default is white).
        bg_color (tuple, optional): The color of the outline (default is black).

    Returns:
        numpy.ndarray: The OpenCV image with the added outlined text.
    """

    img_bg = cv2.putText(img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, font_size, bg_color, thickness * 3, cv2.LINE_AA, False)
    img_fg = cv2.putText(img_bg, text, coords, cv2.FONT_HERSHEY_SIMPLEX, font_size, fg_color, thickness, cv2.LINE_AA, False)
    return img_fg

class ImgPublisher(Node):
    def __init__(self, thread_id, interface: ThreadInterface) -> None:
        """
        This class represents a thread that publishes the frames received from the queue 
        as a ROS image message.

        Args:
            id (int): The ID of the thread
            node (Node): The ROS2 Node object representing the current node
            interface (ThreadInterface): The interface object that contains the queue, mutex, and event
            
        Returns:
            None
        """
        
        # Call the parent constructor
        super().__init__(f'ML_detector_publisher_{thread_id}')
        
        self.id = thread_id
        
        # Access to the corrunt ROS node
        self.logger = self.get_logger()
        
        self.logger.info(f"CONS_{self.id}: Initializing")

        # Debugging options
        self.declare_parameter('debug', 'False')
        
        # Publishing options                  
        self.declare_parameter('rate', '2.0')     
        self.declare_parameter('publish_raw_img', 'False')
        self.declare_parameter('publish_compressed', 'False')
        self.declare_parameter('publish_scaled', 'False')
        self.declare_parameter('publish_scaled_compressed', 'False')
        
        self.declare_parameter('scaled_width', '1920')
        self.declare_parameter('scaled_height', '1080')
        self.declare_parameter('compression_quality', '75')
        
        self.declare_parameter('frame_id_prefix', 'detection_')
        
        # Stability
        self.declare_parameter('no_frame_timeout_sec', '5.0')
        
        # QoS settings       
        self.declare_parameter('img_pub_buffer_depth', 1)
        self.declare_parameter('img_pub_lease_duration', 1)
        self.declare_parameter('img_pub_unreliable', False)
        
        # Timestamps
        self.declare_parameter('add_timestamp', 'True')
        
        # Timestamp position for full resolution image
        self.declare_parameter('time_x', TIME_DEFAULT_X)
        self.declare_parameter('time_y', TIME_DEFAULT_Y)
        self.declare_parameter('time_font_size', TIME_DEFAULT_SIZE)
        self.declare_parameter('time_font_thickness', TIME_DEFAULT_THICKNESS)
        
        # Timestamp position for scaled image
        self.declare_parameter('scaled_time_x', TIME_DEFAULT_X)
        self.declare_parameter('scaled_time_y', TIME_DEFAULT_Y)
        self.declare_parameter('scaled_time_font_size', TIME_DEFAULT_SIZE)
        self.declare_parameter('scaled_time_font_thickness', TIME_DEFAULT_THICKNESS)
            
        # Inter-thread communication
        self.q = interface.get_queue()
        self.mutex = interface.get_lock()
                
        # Read parameters
        
        # Debugging
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # Timing
        rate = self.get_parameter('rate').get_parameter_value().double_value
        
        # Stability
        self.no_frame_timeout_sec = self.get_parameter('no_frame_timeout_sec').get_parameter_value().double_value
        self.no_frame_timeout_en = self.no_frame_timeout_sec > 0.0
        
        # Camera frame
        self.frame_id_prefix = self.get_parameter('frame_id_prefix').get_parameter_value().string_value            
        
        # QoS profile
        qos_buffer_depth = self.get_parameter('img_pub_buffer_depth').get_parameter_value().integer_value
        qos_lease_duration = self.get_parameter('img_pub_lease_duration').get_parameter_value().integer_value
        qos_pub_unreliable = self.get_parameter('img_pub_unreliable').get_parameter_value().bool_value
        
        # Publishing
        self.publish_image_raw = self.get_parameter('publish_raw_img').get_parameter_value().bool_value
        self.publish_compressed = self.get_parameter('publish_compressed').get_parameter_value().bool_value
        self.publish_scaled_image_raw = self.get_parameter('publish_scaled').get_parameter_value().bool_value
        self.publish_scaled_compressed = self.get_parameter('publish_scaled_compressed').get_parameter_value().bool_value
        scaled_width = self.get_parameter('scaled_width').get_parameter_value().integer_value
        scaled_height = self.get_parameter('scaled_height').get_parameter_value().integer_value
        self.compression_quality = self.get_parameter('compression_quality').get_parameter_value().integer_value
        
        # Timestamps
        self.add_timestamp = self.get_parameter('add_timestamp').get_parameter_value().bool_value
        
        # Timestamp position for full resolution image
        self.time_x = self.get_parameter('time_x').get_parameter_value().integer_value
        self.time_y = self.get_parameter('time_y').get_parameter_value().integer_value
        self.time_font_size = self.get_parameter('time_font_size').get_parameter_value().double_value
        self.time_font_thickness = self.get_parameter('time_font_thickness').get_parameter_value().integer_value
        
        # Timestamp position for scaled image
        self.scaled_time_x = self.get_parameter('scaled_time_x').get_parameter_value().integer_value
        self.scaled_time_y = self.get_parameter('scaled_time_y').get_parameter_value().integer_value
        self.scaled_time_font_size = self.get_parameter('scaled_time_font_size').get_parameter_value().double_value
        self.scaled_time_font_thickness = self.get_parameter('scaled_time_font_thickness').get_parameter_value().integer_value
        
        # Time to sleep between iterations to maintain the desired rate
        self.publish_rate = 1.0 / rate
        self.last_publish_time = time.time()
        
        # Frame to publish
        self.frame_old = None
        
        self.scaled_resolution = (scaled_width, scaled_height)
        
        if (scaled_width <= 0 or scaled_height <= 0) and (self.publish_scaled_image_raw or self.publish_scaled_compressed):
            self.logger.warn(f"Requested publishing of scaled image, but scaled resolution is invalid ({scaled_width}, {scaled_height}). Scaling disabled.")
            self.publish_scaled_image_raw = False
            self.publish_scaled_compressed = False
        
        self.logger.info(f"CONS_{self.id}: Publish image_raw: {self.publish_image_raw}")
        self.logger.info(f"CONS_{self.id}: Publish compressed: {self.publish_compressed}")
        self.logger.info(f"CONS_{self.id}: Publish scaled image_raw: {self.publish_scaled_image_raw} ({scaled_width}, {scaled_height})")
        self.logger.info(f"CONS_{self.id}: Publish scaled compressed: {self.publish_scaled_compressed} ({scaled_width}, {scaled_height})")
        
        if not self.publish_image_raw and not self.publish_compressed and not self.publish_scaled_image_raw and not self.publish_scaled_compressed:
            self.logger.fatal("All publishing image types have been disabled, there is nothing to publish!")
            raise RuntimeError("Nothing to publish!")
        
        self.logger.info(f"CONS_{self.id}: FRAME_ID_PREFIX: {self.frame_id_prefix}")
        
        # Error codes
        self.error_code = MLErrorCodes.NO_ERROR
        self.prev_error_code = MLErrorCodes.NO_ERROR
        
        # Timer for sending error codes every 10 seconds
        self.error_timer = self.create_timer(10, self._send_status)
        self.error_timer.reset()
        
        topic_base = interface.get_topic()
        
        # Output topics
        raw_pub_topic = topic_base + '/image_raw'
        compressed_pub_topic = topic_base + '/image_raw/compressed'
        scaled_raw_pub_topic = topic_base + '/scaled/image_raw'
        scaled_compressed_pub_topic = topic_base + '/scaled/image_raw/compressed'
        
        # QoS profiles
        reliable_sensor_qos_profile = create_sensor_qos_profile(qos_buffer_depth, qos_lease_duration)
        pub_qos_profile = qos.qos_profile_sensor_data if qos_pub_unreliable else reliable_sensor_qos_profile
        
        # Setup publishers
        self.info_pub = None
        self.raw_pub = None
        self.compressed_pub = None
        self.scaled_raw_pub = None
        self.scaled_compressed_pub = None
        
        if self.publish_image_raw:
            self.raw_pub = self.create_publisher(Image, raw_pub_topic, 
                                                      qos_profile=pub_qos_profile)
            
            self.logger.info(f"CONS_{self.id}: Publishing to {raw_pub_topic}")
        
        if self.publish_compressed:
            self.compressed_pub = self.create_publisher(CompressedImage, compressed_pub_topic, 
                                                             qos_profile=pub_qos_profile)
            
            self.logger.info(f"CONS_{self.id}: Publishing to {compressed_pub_topic}")

        if self.publish_scaled_image_raw:
            self.scaled_raw_pub = self.create_publisher(Image, scaled_raw_pub_topic, 
                                                             qos_profile=pub_qos_profile)
            
            self.logger.info(f"CONS_{self.id}: Publishing to {scaled_raw_pub_topic}")
        
        if self.publish_scaled_compressed:
            self.scaled_compressed_pub = self.create_publisher(CompressedImage, scaled_compressed_pub_topic, 
                                                                    qos_profile=pub_qos_profile)
            
            self.logger.info(f"CONS_{self.id}: Publishing to {scaled_compressed_pub_topic}")

        # Setup OpenCV bridge for converting images to ROS2 messages
        self.bridge = CvBridge()
        
        # Setup timer for publishing images
        self.send_timer = self.create_timer(self.publish_rate, self._send_frame)
        self.last_pub_time = self.get_clock().now().to_msg()
        
        self.logger.info(f"CONS_{self.id}: Initialization complete")
        
    def _send_status(self) -> None:
        """
        Send an error code to the error code topic.
        
        Args:
            None.
        
        Returns:
            None.
        """
        
        if self.error_code != self.prev_error_code:
            self.logger.info(f"CONS_{self.id}: Error code changed: Current: {self.error_code}, Previous: {self.prev_error_code}")
            self.prev_error_code = self.error_code
        
        # TODO: Send error code to MQTT topic
    
    def _send_frame(self) -> None:
        """
        The main loop of the `OakDImgPublisher` thread. It Reads the camera frames from the queue, 
        then publishes both raw_image and compressed image topics.

        Returns:
            None.
        """

        frame = None
        error_mode = False

        # If it exists, use the old frame by default
        if self.frame_old is not None:
            frame = self.frame_old

        # Get frame from queue
        if not self.q.empty():
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Not empty")

            with self.mutex:
                frame_r = self.q.get()
                
                # Keep the old frame if the new frame is None
                if frame_r is not None:
                    frame = frame_r
                else:
                    error_mode = True
                    self.error_code = MLErrorCodes.ML_ERROR
                    
            # If we got a frame, update the old frame
            if frame is not None:
                # Get current time                    
                timestamp = frame.get_timestamp()
                
                # Update old frame
                self.frame_old = frame
                
                # Update last publish time
                self.last_publish_time = time.time()
                        
        # No frame to publish
        if frame is None:
            self.error_code = MLErrorCodes.ML_ERROR
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Frame is None")
            self.send_timer.reset()
            return
        
        # Get image from frame, make a copy so that we don't modify the original
        img = frame.get_image().copy()
        timestamp = self.get_clock().now().to_msg()
        timestamp_sec = frame.get_timestmp_as_sec()
        frame_id = frame.get_frame_id()
        
        if img is None:
            self.error_code = MLErrorCodes.ML_ERROR
            error_mode = True
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Image in frame is None")
            self.send_timer.reset()
            return
        
        # If we have not received any frames from the camera driver for more than 5 seconds,
        # add an error message to the image
        if self.no_frame_timeout_en and (time.time() - self.last_publish_time > self.no_frame_timeout_sec):
            error_mode = True
        
        # Set error code
        if error_mode:
            self.error_code = MLErrorCodes.ML_ERROR
        else:
            self.error_code = MLErrorCodes.NO_ERROR
        
        if self.debug:
            self.logger.info(f"CONS_{self.id}: Got frame: {img.shape}")
        
        scaled_image = None
        if self.publish_scaled_image_raw or self.publish_scaled_compressed:
            scaled_image = cv2.resize(img, self.scaled_resolution)
        
        if self.add_timestamp:
            ts_str = "CV " + datetime.datetime.fromtimestamp(int(timestamp_sec)).strftime('%Y-%m-%d %H:%M:%S')
            ts_text_size, _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 
                                              self.time_font_size, self.time_font_thickness * 3)
            
            # Draw black solid background for timestamp
            img = cv2.rectangle(img, (self.time_x - 5, self.time_y - ts_text_size[1] - 5), 
                               (self.time_x + ts_text_size[0], self.time_y + 10), (0, 0, 0), -1)
            
            # Draw timestamp
            img = cv_outline_text(img, ts_str, (self.time_x, self.time_y),
                                  self.time_font_size, self.time_font_thickness, 
                                  (255, 255, 255))
            
        
        if error_mode:
            error_txt = "CV ERROR"
            # Put error message on image, black text on red background, centered
            # Text
            text_size, _ = cv2.getTextSize(error_txt, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.time_font_size, self.time_font_thickness)
            text_x = int((img.shape[1] - text_size[0]) / 2)
            text_y = int((img.shape[0] + text_size[1]) / 4)
            img = cv_outline_text(img, error_txt, (text_x, text_y), 
                                            self.time_font_size, self.time_font_thickness,
                                            (0, 0, 0), (0, 0, 255))
            
            # Outline
            img = cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                                            (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), 2)

        # Full size raw image
        if self.publish_image_raw:
            # Convert OpenCV image to ROS2 message
            frame = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            frame.header.frame_id = self.frame_id_prefix + frame_id
            frame.header.stamp = timestamp
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Converted to ROS2 message: {frame.height}x{frame.width}")
        
        # Full size compressed image
        if self.publish_compressed:
            compressed_frame_msg = self.bridge.cv2_to_compressed_imgmsg(img, 'jpeg')
            compressed_frame_msg.header.frame_id = self.frame_id_prefix + frame_id + frame_id
            compressed_frame_msg.header.stamp = timestamp
            
        if (self.publish_scaled_image_raw or self.publish_scaled_compressed) and scaled_image is None:
            raise RuntimeError("Scaled image is None! This is a bug!")
        
        if self.add_timestamp:
            ts_str = "CV " + datetime.datetime.fromtimestamp(int(timestamp_sec)).strftime('%Y-%m-%d %H:%M:%S')
            ts_text_size, _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 
                                              self.scaled_time_font_size, self.scaled_time_font_thickness * 3)
            
            # Draw black solid background for timestamp
            scaled_image = cv2.rectangle(scaled_image, (0, 0), (self.scaled_time_x + ts_text_size[0], self.scaled_time_y + 5), 
                                         (0, 0, 0), -1)
            
            # Draw timestamp
            scaled_image = cv_outline_text(scaled_image, ts_str, (self.scaled_time_x, self.scaled_time_y),
                                           self.scaled_time_font_size, self.scaled_time_font_thickness, 
                                           (255, 255, 255))
        
        if (self.publish_scaled_image_raw or self.publish_scaled_compressed) and error_mode:
            error_txt = "CV ERROR"
            # Put error message on image, black text on red background, centered
            # Text
            text_size, _ = cv2.getTextSize(error_txt, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.scaled_time_font_size, self.scaled_time_font_thickness)
            text_x = int((scaled_image.shape[1] - text_size[0]) / 2)
            text_y = int((scaled_image.shape[0] + text_size[1]) / 4)
            scaled_image = cv_outline_text(scaled_image, error_txt, (text_x, text_y), 
                                            self.scaled_time_font_size, self.scaled_time_font_thickness, 
                                            (0, 0, 0), (0, 0, 255))
            
            # Outline
            scaled_image = cv2.rectangle(scaled_image, (text_x - 5, text_y - text_size[1] - 5), 
                                            (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), 2)
        
        if self.publish_scaled_image_raw:
            # Convert scaled OpenCV image to ROS2 message
            scaled_frame = self.bridge.cv2_to_imgmsg(scaled_image, 'bgr8')
            
            scaled_frame.header.frame_id = self.frame_id_prefix + frame_id
            scaled_frame.header.stamp = timestamp
            
        if self.publish_scaled_compressed:
            compressed_scaled_frame_msg = self.bridge.cv2_to_compressed_imgmsg(scaled_image, 'jpeg')
            compressed_scaled_frame_msg.header.frame_id = self.frame_id_prefix + frame_id
            compressed_scaled_frame_msg.header.stamp = timestamp
            
        # Publish topics. All publishing have been moved down here 
        # so that we can publish them at more or less the same time
        
        if self.debug:
            self.logger.info(f"CONS_{self.id}: Publishing topics")
        
        # Publish full size raw image
        if self.publish_image_raw:
            if self.raw_pub is None:
                raise RuntimeError("self.raw_pub is None! This is a bug!")
            self.raw_pub.publish(frame)
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Publishing raw image")
        
        # Publish full size compressed image
        if self.publish_compressed:
            if self.compressed_pub is None:
                raise RuntimeError("self.compressed_pub is None! This is a bug!")
            self.compressed_pub.publish(compressed_frame_msg)
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Publishing compressed image")

        # Publish scaled raw image
        if self.publish_scaled_image_raw:
            if self.scaled_raw_pub is None:
                raise RuntimeError("self.scaled_raw_pub is None! This is a bug!")
            self.scaled_raw_pub.publish(scaled_frame)
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Publishing scaled raw image")
        
        # Publish scaled compressed image
        if self.publish_scaled_compressed:
            if self.scaled_compressed_pub is None:
                raise RuntimeError("self.scaled_compressed_pub is None! This is a bug!")
            self.scaled_compressed_pub.publish(compressed_scaled_frame_msg)
            
            if self.debug:
                self.logger.info(f"CONS_{self.id}: Publishing scaled compressed image")

        # Log the time it took to publish the frame
        if self.debug:
            end_time = self.get_clock().now().to_msg()
            looptime = end_time.sec - timestamp.sec + (end_time.nanosec - timestamp.nanosec) / 1e9
            pub_hz = end_time.sec - self.last_pub_time.sec + (end_time.nanosec - self.last_pub_time.nanosec) / 1e9
            
            self.last_pub_time = end_time
            self.logger.info(f"CONS_{self.id}: END_OF_LOOP {looptime}, rate: {1.0 / pub_hz}")
        
        # Sleep for the remaining time to maintain the desired publishing rate
        self.send_timer.reset()
