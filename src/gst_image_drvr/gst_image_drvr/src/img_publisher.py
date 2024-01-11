import os
import cv2
import yaml
import time
import datetime
import numpy as np
from queue import Queue
from threading import Thread, Lock, Event

import rclpy
import rclpy.qos as qos
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

class CameraErrorCodes:
    NO_ERROR = 0
    CAMERA_ERROR = 1

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

class GstImgPublisher(Thread):
    def __init__(self, node: Node, mutex: Lock, q: Queue, img_event: Event) -> None:
        """
        This class represents a thread that publishes the frames received from the queue 
        as a ROS image message.

        Args:
            node (Node): The ROS2 Node object representing the current node
            mutex (Lock): The mutex object for thread-safe access to the queue
            q (Queue): The queue object for reading frames from the camera driver thread
        """
        
        # Call the parent constructor
        super().__init__()
        
        # Access to the corrunt ROS node
        self.node = node
        self.logger = node.get_logger()

        # Inter-thread communication
        self.q = q
        self.mutex = mutex
        self.img_event = img_event
        
        self.should_quit = False
                
        # Read parameters
        
        # Debugging
        self.debug = self.node.get_parameter('debug').get_parameter_value().bool_value
        
        # Timing
        rate = self.node.get_parameter('rate').get_parameter_value().double_value
        
        # Stability
        self.no_frame_timeout_sec = self.node.get_parameter('no_frame_timeout_sec').get_parameter_value().double_value
        self.no_frame_timeout_en = self.no_frame_timeout_sec > 0.0
        
        # Camera frame
        self.add_timestamp = self.node.get_parameter('add_timestamp').get_parameter_value().bool_value
        self.frame_id = self.node.get_parameter('frame_id').get_parameter_value().string_value            
        camera_info_file = self.node.get_parameter('camera_info').get_parameter_value().string_value
        
        # QoS profile
        qos_buffer_depth = self.node.get_parameter('buffer_depth').get_parameter_value().integer_value
        qos_lease_duration = self.node.get_parameter('lease_duration').get_parameter_value().integer_value
        qos_pub_unreliable = self.node.get_parameter('pub_unreliable').get_parameter_value().bool_value
        
        # Publishing
        self.publish_image_raw = self.node.get_parameter('publish_image_raw').get_parameter_value().bool_value
        self.publish_compressed = self.node.get_parameter('publish_compressed').get_parameter_value().bool_value
        self.publish_scaled_image_raw = self.node.get_parameter('publish_scaled_image_raw').get_parameter_value().bool_value
        self.publish_scaled_compressed = self.node.get_parameter('publish_scaled_compressed').get_parameter_value().bool_value
        scaled_width = self.node.get_parameter('scaled_width').get_parameter_value().integer_value
        scaled_height = self.node.get_parameter('scaled_height').get_parameter_value().integer_value
        self.compression_quality = self.node.get_parameter('compression_quality').get_parameter_value().integer_value
        
        # Timestamp position for full resolution image
        self.time_x = self.node.get_parameter('time_x').get_parameter_value().integer_value
        self.time_y = self.node.get_parameter('time_y').get_parameter_value().integer_value
        self.time_font_size = self.node.get_parameter('time_font_size').get_parameter_value().double_value
        self.time_font_thickness = self.node.get_parameter('time_font_thickness').get_parameter_value().integer_value
        
        # Timestamp position for scaled image
        self.scaled_time_x = self.node.get_parameter('scaled_time_x').get_parameter_value().integer_value
        self.scaled_time_y = self.node.get_parameter('scaled_time_y').get_parameter_value().integer_value
        self.scaled_time_font_size = self.node.get_parameter('scaled_time_font_size').get_parameter_value().double_value
        self.scaled_time_font_thickness = self.node.get_parameter('scaled_time_font_thickness').get_parameter_value().integer_value
        
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
        
        self.logger.info(f"Publish image_raw: {self.publish_image_raw}")
        self.logger.info(f"Publish compressed: {self.publish_compressed}")
        self.logger.info(f"Publish scaled image_raw: {self.publish_scaled_image_raw} ({scaled_width}, {scaled_height})")
        self.logger.info(f"Publish scaled compressed: {self.publish_scaled_compressed} ({scaled_width}, {scaled_height})")
        
        if not self.publish_image_raw and not self.publish_compressed and not self.publish_scaled_image_raw and not self.publish_scaled_compressed:
            self.logger.fatal("All publishing image types have been disabled, there is nothing to publish!")
            raise RuntimeError("Nothing to publish!")
        
        self.cam_info = self._get_camera_info(camera_info_file)
        self.logger.info("FRAME_ID: " + self.frame_id)
        
        # Error codes
        self.error_code = CameraErrorCodes.NO_ERROR
        self.prev_error_code = CameraErrorCodes.NO_ERROR
        
        # Timer for sending error codes every 10 seconds
        self.error_timer = self.node.create_timer(10, self._send_status)
        self.error_timer.reset()
        
        # Current node's namespace
        ns = self.node.get_namespace()
        
        if ns == '/':
            ns = ''
        
        # Output topics
        raw_pub_topic = ns + '/' + self.frame_id + '/image_raw'
        info_pub_topic = ns + '/' + self.frame_id + '/camera_info'
        compressed_pub_topic = ns + '/' + self.frame_id + '/image_raw/compressed'
        scaled_raw_pub_topic = ns + '/' + self.frame_id + '/scaled/image_raw'
        scaled_compressed_pub_topic = ns + '/' + self.frame_id + '/scaled/image_raw/compressed'
        
        # QoS profiles
        reliable_sensor_qos_profile = self._create_sensor_qos_profile(qos_buffer_depth, qos_lease_duration)
        pub_qos_profile = qos.qos_profile_sensor_data if qos_pub_unreliable else reliable_sensor_qos_profile
        
        # Setup publishers
        self.info_pub = None
        self.raw_pub = None
        self.compressed_pub = None
        self.scaled_raw_pub = None
        self.scaled_compressed_pub = None
        
        if self.publish_image_raw or self.publish_compressed:
            self.info_pub = self.node.create_publisher(CameraInfo, info_pub_topic, 
                                                       qos_profile=pub_qos_profile)
            
            self.logger.info(f"Publishing to {info_pub_topic}")
        
        if self.publish_image_raw:
            self.raw_pub = self.node.create_publisher(Image, raw_pub_topic, 
                                                      qos_profile=pub_qos_profile)
            
            self.logger.info(f"Publishing to {raw_pub_topic}")
        
        if self.publish_compressed:
            self.compressed_pub = self.node.create_publisher(CompressedImage, compressed_pub_topic, 
                                                             qos_profile=pub_qos_profile)
            
            self.logger.info(f"Publishing to {compressed_pub_topic}")

        if self.publish_scaled_image_raw:
            self.scaled_raw_pub = self.node.create_publisher(Image, scaled_raw_pub_topic, 
                                                             qos_profile=pub_qos_profile)
            
            self.logger.info(f"Publishing to {scaled_raw_pub_topic}")
        
        if self.publish_scaled_compressed:
            self.scaled_compressed_pub = self.node.create_publisher(CompressedImage, scaled_compressed_pub_topic, 
                                                                    qos_profile=pub_qos_profile)
            
            self.logger.info(f"Publishing to {scaled_compressed_pub_topic}")

        # For publishing OpenCV topics
        self.bridge = CvBridge()

    def _create_sensor_qos_profile(self, buffer_depth: int, lease_duration: int) -> qos.QoSProfile:
        """
        Create and return a QoS profile for the sensor data.

        Args:
            None.

        Returns:
            QoSProfile: The QoS profile for the sensor data.
        """
        
        # Create the QoS profile with the desired settings
        qos_profile = qos.QoSProfile(depth=buffer_depth)
        
        # Set the reliability to reliable to ensure message delivery
        qos_profile.reliability = qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE
        
        # Set the durability to transient local to allow late-joining subscribers to receive the last message
        qos_profile.durability = qos.QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL
        
        # Set the liveliness to automatic with a lease duration
        qos_profile.liveliness = qos.QoSLivelinessPolicy.RMW_QOS_POLICY_LIVELINESS_AUTOMATIC
        qos_profile.liveliness_lease_duration = rclpy.time.Duration(seconds=lease_duration)
        
        qos_profile.avoid_ros_namespace_conventions = False

        return qos_profile


    def _get_camera_info(self, calib_file: str) -> CameraInfo:
        """
        This method reads the ROS camera calibration (camera info) file and returns a
        CameraInfo object containing the file.

        Args:
            calib_file (str): The path to the ROS camera calibration (camera info) file.

        Returns:
            CameraInfo: A CameraInfo object containing the calibration information from
            the specified file.

        Raises:
            FileNotFoundError: If the specified file cannot be found.
            yaml.YAMLError: If there is an error in parsing the YAML file.
        """
        
        self.logger.info("Camera info extraction")
            
        cam_info = CameraInfo()

        if calib_file == '':
            self.logger.warn("No camera info file specified! Publishing empty camera info")
        
        else:
            calib_file = os.path.abspath(os.path.expanduser(calib_file))

            try:
                with open(calib_file, 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)

                    cam_info.height = params['image_height']
                    cam_info.width = params['image_width']
                    cam_info.distortion_model = params['distortion_model']
                    cam_info.k = params['camera_matrix']['data']
                    cam_info.d = params['distortion_coefficients']['data']
                    cam_info.r = params['rectification_matrix']['data']
                    cam_info.p = params['projection_matrix']['data']

            except (yaml.YAMLError, OSError) as err:
                raise RuntimeError(f"Cannot read camera calibration file: {err}")

        return cam_info
    
    def stop(self) -> None:
        """
        Signal the camera driver thread to stop publishing and exit gracefully.

        Sets the `should_quit` flag to `True` and logs a message indicating that 
        the stop signal has been received.

        Returns:
            None.
        """
        
        self.should_quit = True
        self.logger.info("GST Publisher: Stopping consumer thread")
    
    def _sleep_remaining_time(self) -> None:
        """
        Sleep for the remaining time to maintain the desired publishing rate.
        
        Args:
            None
            
        Returns:
            None
        """
        
        sleep_time = 0.0
        current_time = time.time()
        
        loop_duration = current_time - self.last_publish_time
        
        # Generate a proper publish rate. 
        # If the loop duration is less than the desired publish rate, we are in error mode
        if loop_duration < self.publish_rate:
            sleep_time = self.publish_rate - loop_duration
        else:
            sleep_time = self.publish_rate
        
        time_slept = 0.0
        
        # Sleep in small increments to allow for quick exit
        # 0.03 seconds still allows for 30 FPS
        time_to_sleep = 0.03
        while time_slept < sleep_time and not self.img_event.is_set() and not self.should_quit:
            time_slept += time_to_sleep
            time.sleep(time_to_sleep)
        
    def _send_status(self) -> None:
        """
        Send an error code to the error code topic.
        
        Args:
            None.
        
        Returns:
            None.
        """
        
        if self.error_code != self.prev_error_code:
            self.logger.info(f"Error code changed: Current: {self.error_code}, Previous: {self.prev_error_code}")
            self.prev_error_code = self.error_code
        
        # TODO: Send error code to MQTT topic
    
    def run(self) -> None:
        """
        The main loop of the `OakDImgPublisher` thread. It Reads the camera frames from the queue, 
        then publishes both raw_image and compressed image topics.

        Returns:
            None.
        """

        self.logger.info("GST Publisher: Running main loop")
        while not self.should_quit:
            frame = None
            error_mode = False

            # If it exists, use the old frame by default
            if self.frame_old is not None:
                frame = self.frame_old

            # Get frame from queue
            if not self.q.empty() and self.img_event.is_set():
                if self.debug:
                    self.logger.info("CONS: Not empty")

                with self.mutex:
                    frame_r = self.q.get()
                    
                    # Keep the old frame if the new frame is None
                    if frame_r is not None:
                        frame = frame_r
                    else:
                        error_mode = True
                        self.error_code = CameraErrorCodes.CAMERA_ERROR
                        
                    self.img_event.clear()

                # If we got a frame, update the old frame
                if frame is not None:
                    # Get current time                    
                    timestamp = frame.get_timestamp()
                    timestamp_sec = frame.get_timestmp_as_sec()
                    
                    # Update old frame
                    self.frame_old = frame
                    
                    # Update last publish time
                    self.last_publish_time = time.time()
                            
            # No frame to publish
            if frame is None:
                self.error_code = CameraErrorCodes.CAMERA_ERROR
                self._sleep_remaining_time()
                
                if self.debug:
                    self.logger.info("CONS: Frame is None")
                continue
            
            # Get image from frame, make a copy so that we don't modify the original
            img = frame.get_image().copy()
            
            if img is None:
                self.error_code = CameraErrorCodes.CAMERA_ERROR
                error_mode = True
                self._sleep_remaining_time()
                
                if self.debug:
                    self.logger.info("CONS: Image in frame is None")
                continue
            
            # If we have not received any frames from the camera driver for more than 5 seconds,
            # add an error message to the image
            if self.no_frame_timeout_en and (time.time() - self.last_publish_time > self.no_frame_timeout_sec):
                error_mode = True
            
            # Set error code
            if error_mode:
                self.error_code = CameraErrorCodes.CAMERA_ERROR
            else:
                self.error_code = CameraErrorCodes.NO_ERROR
            
            if self.debug:
                self.logger.info(f"CONS: Got frame: {img.shape}")
            
            scaled_image = None
            if self.publish_scaled_image_raw or self.publish_scaled_compressed:
                scaled_image = cv2.resize(img, self.scaled_resolution)
            
            if self.add_timestamp:
                # Put timestamp on image
                ts_str = datetime.datetime.fromtimestamp(int(timestamp_sec)).strftime('%Y-%m-%d %H:%M:%S')
                img = cv_outline_text(img, ts_str, (self.time_x, self.time_y), 
                                                self.time_font_size, self.time_font_thickness)
                
                if error_mode:
                    error_txt = "CAMERA ERROR"
                    # Put error message on image, black text on red background, centered
                    # Text
                    text_size, _ = cv2.getTextSize(error_txt, cv2.FONT_HERSHEY_SIMPLEX, 
                                                   self.time_font_size, self.time_font_thickness)
                    text_x = int((img.shape[1] - text_size[0]) / 2)
                    text_y = int((img.shape[0] + text_size[1]) / 4)
                    img = cv_outline_text(img, error_txt, (text_x, text_y), 
                                                    self.time_font_size, self.time_font_thickness, (0, 0, 0), (0, 0, 255))
                    
                    # Outline
                    img = cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                                                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), 2)

            # Full size raw image
            if self.publish_image_raw:
                # Convert OpenCV image to ROS2 message
                frame = self.bridge.cv2_to_imgmsg(img, 'bgr8')
                frame.header.frame_id = self.frame_id
                frame.header.stamp = timestamp
                if self.debug:
                    self.logger.info(f"CONS: Converted to ROS2 message: {frame.height}x{frame.width}")
            
            # Full size compressed iamge
            if self.publish_compressed:
                compressed_frame_msg = self.bridge.cv2_to_compressed_imgmsg(img, 'jpeg')
                compressed_frame_msg.header.frame_id = self.frame_id
                compressed_frame_msg.header.stamp = timestamp
            
            # Publish camera info only when we are publishing the full size image
            if self.publish_image_raw or self.publish_compressed:
                # Camera info
                if self.cam_info.width == 0:
                    self.cam_info.width = img.shape[1]
                
                if self.cam_info.height == 0:
                    self.cam_info.height = img.shape[0]
                    
                self.cam_info.header.frame_id = self.frame_id
                self.cam_info.header.stamp = timestamp
                
            if (self.publish_scaled_image_raw or self.publish_scaled_compressed) and scaled_image is None:
                raise RuntimeError("Scaled image is None! This is a bug!")
            
            if (self.publish_scaled_image_raw or self.publish_scaled_compressed) and self.add_timestamp:
                # Put timestamp on image
                ts_str = datetime.datetime.fromtimestamp(int(timestamp_sec)).strftime('%Y-%m-%d %H:%M:%S')
                scaled_image = cv_outline_text(scaled_image, ts_str, (self.scaled_time_x, self.scaled_time_y), 
                                               self.scaled_time_font_size, self.scaled_time_font_thickness)
                
                if error_mode:
                    error_txt = "CAMERA ERROR"
                    # Put error message on image, black text on red background, centered
                    # Text
                    text_size, _ = cv2.getTextSize(error_txt, cv2.FONT_HERSHEY_SIMPLEX, 
                                                   self.scaled_time_font_size, self.scaled_time_font_thickness)
                    text_x = int((scaled_image.shape[1] - text_size[0]) / 2)
                    text_y = int((scaled_image.shape[0] + text_size[1]) / 4)
                    scaled_image = cv_outline_text(scaled_image, error_txt, (text_x, text_y), 
                                                    self.scaled_time_font_size, self.scaled_time_font_thickness, (0, 0, 0), (0, 0, 255))
                    
                    # Outline
                    scaled_image = cv2.rectangle(scaled_image, (text_x - 5, text_y - text_size[1] - 5), 
                                                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), 2)
            
            if self.publish_scaled_image_raw:
                # Convert scaled OpenCV image to ROS2 message
                scaled_frame = self.bridge.cv2_to_imgmsg(scaled_image, 'bgr8')
                
                scaled_frame.header.frame_id = self.frame_id
                scaled_frame.header.stamp = timestamp
                
            if self.publish_scaled_compressed:
                # Compressed image:
                compressed_scaled_frame_msg = self.bridge.cv2_to_compressed_imgmsg(scaled_image, 'jpeg')
                compressed_scaled_frame_msg.header.frame_id = self.frame_id
                compressed_scaled_frame_msg.header.stamp = timestamp
                
            # Publish topics. All publishing have been moved down here 
            # so that we can publish them at more or less the same time
            
            if self.debug:
                self.logger.info("CONS: Publishing topics")
            
            # Publish camera info only when we are publishing the full size image
            if self.publish_image_raw or self.publish_compressed:
                if self.info_pub is None:
                    raise RuntimeError("self.info_pub is None! This is a bug!")
                
                # Publish camera info
                self.info_pub.publish(self.cam_info)
                
                if self.debug:
                    self.logger.info(f"CONS: Publishing camera info: {self.cam_info.width}x{self.cam_info.height}")
                    self.logger.info("CONS: Publishing camera info")
            
            # Publish full size raw image
            if self.publish_image_raw:
                if self.raw_pub is None:
                    raise RuntimeError("self.raw_pub is None! This is a bug!")
                self.raw_pub.publish(frame)
                
                if self.debug:
                    self.logger.info("CONS: Publishing raw image")
            
            # Publish full size compressed image
            if self.publish_compressed:
                if self.compressed_pub is None:
                    raise RuntimeError("self.compressed_pub is None! This is a bug!")
                self.compressed_pub.publish(compressed_frame_msg)
                
                if self.debug:
                    self.logger.info("CONS: Publishing compressed image")

            # Publish scaled raw image
            if self.publish_scaled_image_raw:
                if self.scaled_raw_pub is None:
                    raise RuntimeError("self.scaled_raw_pub is None! This is a bug!")
                self.scaled_raw_pub.publish(scaled_frame)
                
                if self.debug:
                    self.logger.info("CONS: Publishing scaled raw image")
            
            # Publish scaled compressed image
            if self.publish_scaled_compressed:
                if self.scaled_compressed_pub is None:
                    raise RuntimeError("self.scaled_compressed_pub is None! This is a bug!")
                self.scaled_compressed_pub.publish(compressed_scaled_frame_msg)
                
                if self.debug:
                    self.logger.info("CONS: Publishing scaled compressed image")
                        
            # Sleep to maintain the desired publishing rate
            self._sleep_remaining_time()
