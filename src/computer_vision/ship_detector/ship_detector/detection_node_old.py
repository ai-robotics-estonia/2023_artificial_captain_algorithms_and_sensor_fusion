#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
import rclpy.qos as qos

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from turbojpeg import TurboJPEG, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE

from src.common import ThreadInterface, Frame
from src.img_publisher import ImgPublisher
from src.object_detection import ObjectDetection

from detection_interface.msg import BoundingBox, BoundingBoxes

class YoloDetector():
    def __init__(self, node: Node) -> None:
        
        self.node = node
        self.logger = node.get_logger()        
        
        ################################
        # ROS 2 parameter declarations #
        ################################
        
        # Enabling different components of the node
        
        # Debugging options
        self.node.declare_parameter('debug', 'False')
        
        # Publishing options                  
        self.node.declare_parameter('rate', '2.0')     
        self.node.declare_parameter('publish_raw_img', 'False')
        self.node.declare_parameter('publish_compressed', 'False')
        self.node.declare_parameter('publish_scaled', 'False')
        self.node.declare_parameter('publish_scaled_compressed', 'False')
        
        self.node.declare_parameter('scaled_width', '1920')
        self.node.declare_parameter('scaled_height', '1080')
        self.node.declare_parameter('compression_quality', '75')
        
        self.node.declare_parameter('frame_id_prefix', 'detection_')
        
        # Stability
        self.node.declare_parameter('no_frame_timeout_sec', '5.0')

        # YOLO config
        self.node.declare_parameter('yolo_model')
        self.node.declare_parameter('input_size', '1280')
        self.node.declare_parameter('confidence_threshold', '0.5')
        self.node.declare_parameter('iou_threshold', '0.5')
        
        # Tracker config
        self.node.declare_parameter('enable_tracker', 'False')
        self.node.declare_parameter('tracker_type', 'bytetrack')
        
        # QoS settings
        self.node.declare_parameter('sub_buffer_depth', 1)
        self.node.declare_parameter('sub_lease_duration', 1)
        self.node.declare_parameter('sub_unreliable', False)
        
        self.node.declare_parameter('img_pub_buffer_depth', 1)
        self.node.declare_parameter('img_pub_lease_duration', 1)
        self.node.declare_parameter('img_pub_unreliable', False)
        
        self.node.declare_parameter('bbox_pub_buffer_depth', 1)
        self.node.declare_parameter('bbox_pub_lease_duration', 1)
        self.node.declare_parameter('bbox_pub_unreliable', False)
        
        # Output config
        self.node.declare_parameter('enable_bounding_boxes', 'True')
        self.node.declare_parameter('bbox_topic', 'bounding_boxes')
        
        # Camera config
        self.node.declare_parameter('num_cams', '1')
        
        #############################
        # Retrieve ROS 2 parameters #
        #############################
        
        # Debugging options
        self.debug = self.node.get_parameter('debug').get_parameter_value().bool_value
        
        # Publishing options
        raw_img_publish = self.node.get_parameter('publish_raw_img').get_parameter_value().bool_value
        compressed_img_publish = self.node.get_parameter('publish_compressed').get_parameter_value().bool_value
        publish_scaled = self.node.get_parameter('publish_scaled').get_parameter_value().bool_value
        publish_scaled_compressed = self.node.get_parameter('publish_scaled_compressed').get_parameter_value().bool_value
        
        # YOLO config
        self.yolo_model = self.node.get_parameter('yolo_model').get_parameter_value().string_value
        self.input_size = self.node.get_parameter('input_size').get_parameter_value().integer_value
        self.conf_thresh = self.node.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_thresh = self.node.get_parameter('iou_threshold').get_parameter_value().double_value
        
        # Tracker config
        self.enable_tracker = self.node.get_parameter('enable_tracker').get_parameter_value().bool_value
        self.tracker_type = self.node.get_parameter('tracker_type').get_parameter_value().string_value
        
        # QoS settings
        sub_buffer_depth = self.node.get_parameter('sub_buffer_depth').get_parameter_value().integer_value
        sub_lease_duration = self.node.get_parameter('sub_lease_duration').get_parameter_value().integer_value
        sub_unreliable = self.node.get_parameter('sub_unreliable').get_parameter_value().bool_value
        
        bbox_pub_buffer_depth = self.node.get_parameter('bbox_pub_buffer_depth').get_parameter_value().integer_value
        bbox_pub_lease_duration = self.node.get_parameter('bbox_pub_lease_duration').get_parameter_value().integer_value
        bbox_pub_unreliable = self.node.get_parameter('bbox_pub_unreliable').get_parameter_value().bool_value
        
        # Output config
        self.enable_bounding_boxes = self.node.get_parameter('enable_bounding_boxes').get_parameter_value().bool_value
        bbox_topic = self.node.get_parameter('bbox_topic').get_parameter_value().string_value
        
        # Camera config
        num_cams = self.node.get_parameter('num_cams').get_parameter_value().integer_value
        
        self.logger.info(f'Number of images: {num_cams}')
        
        # Declare camera topics
        for i in range(num_cams):
            self.logger.info(f'Declaring: cam_{i}_topic')
            self.node.declare_parameter(f'cam_{i}_topic', '')
            
        # Get camera topics
        cams_topics = []
        for i in range(num_cams):
            cams_topics.append(self.node.get_parameter(f'cam_{i}_topic').get_parameter_value().string_value)
        
        for i, topic in enumerate(cams_topics):
            self.logger.info(f'Camera topic: {topic}')
            if topic == '':
                self.logger.warn(f'Camera topic {i} is empty! Please check the launch file.')
                raise ValueError(f'Camera topic {i} is empty! Please check the launch file.')
        
        # QoS settings
        reliable_img_sub_qos_profile = self._create_sensor_qos_profile(sub_buffer_depth, sub_lease_duration)
        reliable_bbox_pub_qos_profile = self._create_sensor_qos_profile(bbox_pub_buffer_depth, bbox_pub_lease_duration)
        
        img_sub_qos_profile = qos.qos_profile_sensor_data if sub_unreliable else reliable_img_sub_qos_profile
        bbox_pub_qos_profile = qos.qos_profile_sensor_data if bbox_pub_unreliable else reliable_bbox_pub_qos_profile
    
        publish_img = raw_img_publish or compressed_img_publish or publish_scaled or publish_scaled_compressed
    
        # Create thread interfaces
        sub_ifs = []
        pub_ifs = []
        
        self.sub_threads = []
        self.pub_threads = []
        
        for i, topic in enumerate(cams_topics):
            sub_if = ThreadInterface(i, topic)
            sub_ifs.append(sub_if)
            
            if publish_img:
                pub_topic = topic.replace('/compressed', '')
                pub_topic = pub_topic.replace('/image_raw', '')
                pub_topic += '/detection'
                
                pub_if = ThreadInterface(i, pub_topic)
                pub_ifs.append(pub_if)
        
        # Create image publisher threads
        if publish_img:
            for i, topic in enumerate(cams_topics):
                pub_thread = ImgPublisher(i, self.node, pub_ifs[i])
                self.pub_threads.append(pub_thread)
        
        # Start image publisher threads
        for thread in self.pub_threads:
            thread.start()
        
        return
    
        self.img_subscibers = []
        self.img_publishers = []
        self.img_scaled_publishers = []
        self.img_compressed_publishers = []
        self.img_scaled_compressed_publishers = []
        self.bbox_publishers = []
        
        for i, topic in enumerate(cams_topics):
            # Create image subscriber for the current camera
            sub_topic_type = CompressedImage if topic.endswith('/compressed') else Image
            sub_cb = self._compressed_img_callback if topic.endswith('/compressed') else self._img_callback
            sub = self.node.create_subscription(sub_topic_type, topic, sub_cb, qos_profile=img_sub_qos_profile)
            self.logger.info(f"Subscribing to {topic}")
            self.img_subscibers.append(sub)
            
            topic_base = topic.replace('/compressed', '')
            topic_base = topic_base.replace('/image_raw', '')
            topic_base += '/detection'
            
            # Create image publishers for the current camera           
            if self.raw_img_publish:
                raw_pub_topic = f'{topic_base}/image_raw'
                raw_pub = self.node.create_publisher(Image, raw_pub_topic, qos_profile=img_pub_qos_profile)
                
                self.logger.info(f"Publishing to {raw_pub_topic}")
                self.img_publishers.append(raw_pub)
            
            if self.compressed_img_publish:
                compressed_pub_topic = f'{topic_base}/image_raw/compressed'
                compressed_pub = self.node.create_publisher(CompressedImage, compressed_pub_topic,
                                                            qos_profile=img_pub_qos_profile)
                
                self.logger.info(f"Publishing to {compressed_pub_topic}")
                self.img_compressed_publishers.append(compressed_pub)

            if self.publish_scaled:
                scaled_raw_pub_topic = f'{topic_base}/scaled/image_raw'
                scaled_raw_pub = self.node.create_publisher(Image, scaled_raw_pub_topic, qos_profile=img_pub_qos_profile)
                
                self.logger.info(f"Publishing to {scaled_raw_pub_topic}")
                self.img_scaled_publishers.append(scaled_raw_pub)
            
            if self.publish_scaled_compressed:
                scaled_compressed_pub_topic = f'{topic_base}/scaled/image_raw/compressed'
                scaled_compressed_pub = self.node.create_publisher(CompressedImage, scaled_compressed_pub_topic,
                                                                   qos_profile=img_pub_qos_profile)
                
                self.logger.info(f"Publishing to {scaled_compressed_pub_topic}")
                self.img_scaled_compressed_publishers.append(scaled_compressed_pub)
                
            # Create bounding box publisher for the current camera
            if self.enable_bounding_boxes:
                bbox_pub_topic = f'{topic_base}/{bbox_topic}'
                bbox_pub = self.node.create_publisher(BoundingBoxes, bbox_pub_topic, qos_profile=bbox_pub_qos_profile)
                
                self.logger.info(f"Publishing to {bbox_pub_topic}")
                self.bbox_publishers.append(bbox_pub)
                
        return
        
        
        
        # Define classes for object detection
        classes = [0,1] #####################################33
        
        # Initialize object detection with YOLO model and parameters
        self.OD = ObjectDetection(yolo_model, input_size, conf_thresh, iou_thresh, tracker, classes)

        # Initialize video processing tools
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.compression_quality = 75 #####################################
        
    def _create_sensor_qos_profile(self, buffer_depth, lease_duration) -> qos.QoSProfile:
        """
        Create and return a QoS profile for the sensor data.

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

    def _compressed_img_callback(self, msg: CompressedImage) -> None:
        self.logger.info(f'Compressed image received')
    
    def _img_callback(self, msg: Image) -> None:
        self.logger.info(f'Image received')
    
    def image_callback(self, cv_image: np.ndarray) -> None:
        """
        Callback function for processing the image.
        
        Arguments:
            cv_image (numpy.ndarray): The input image in OpenCV format.
        
        Returns:
            tuple: A tuple containing the bounding boxes and the processed image.
        """
        
        # Perform object detection using YOLO
        class_ids, conf_vals, bboxes_coords, num_detections, yolo_elapsed_time, res_plot = self.OD.detect_objects(cv_image, False)
        
        # Create a BoundingBoxes message to store the detected bounding boxes
        bboxes_pub = BoundingBoxes()
        bboxes = []

        # Process each detected object
        for i in range(num_detections):
            # Create a BoundingBox message for the current object
            bbox_info = BoundingBox()
            bbox_info.id = i
            bbox_info.class_id = int(class_ids[i])  # ID of the detected object class
            bbox_info.probability = float(conf_vals[i])  # Confidence score for the detection
                
            # Extract the bounding box coordinates
            bbox_info.xmin = int(bboxes_coords[i][0])  # x-coordinate of the top-left corner
            bbox_info.ymin = int(bboxes_coords[i][1])  # y-coordinate of the top-left corner
            bbox_info.xmax = int(bboxes_coords[i][2])  # x-coordinate of the bottom-right corner
            bbox_info.ymax = int(bboxes_coords[i][3])  # y-coordinate of the bottom-right corner

            # Add the bounding box to the list
            bboxes.append(bbox_info)

        # Assign the list of bounding boxes to the BoundingBoxes message
        bboxes_pub.bounding_boxes = bboxes
        
        # Create a Header for the BoundingBoxes message
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        bboxes_pub.header = h

        # Return the BoundingBoxes message and the processed image
        return bboxes_pub, res_plot
    
    def forward(self, msg: CompressedImage) -> None:
        """
        Handles the forward processing of the received message.
        
        Arguments:
            msg (CompressedImage): The received image message.
        """
        
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        bboxes_pub, res_plot = self.image_callback(cv_image)
        self.bbox_forward_publisher.publish(bboxes_pub)
        
        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # No publishing options enabled, do nothing
            pass

        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_forward_scaled_img_publisher.publish(scaled_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_forward_comp_img_publisher.publish(compressed_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish compressed and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_forward_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw image
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_forward_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_forward_scaled_img_publisher.publish(scaled_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw and compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_forward_comp_img_publisher.publish(compressed_image_msg)
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_forward_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw, compressed, and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_forward_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)
            self.det_forward_scaled_img_publisher.publish(scaled_image_msg)
            
        
    def starboard(self, msg: CompressedImage) -> None:
        """
        Handles the starboard processing of the received message.
        
        Arguments:
            msg (CompressedImage): The received image message.
        """
        
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        bboxes_pub, res_plot = self.image_callback(cv_image)
        self.bbox_starboard_publisher.publish(bboxes_pub)
        
        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # No publishing options enabled, do nothing
            pass

        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_starboard_scaled_img_publisher.publish(scaled_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_starboard_comp_img_publisher.publish(compressed_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish compressed and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_starboard_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw image
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_starboard_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_starboard_scaled_img_publisher.publish(scaled_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw and compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_starboard_comp_img_publisher.publish(compressed_image_msg)
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_starboard_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw, compressed, and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_starboard_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)
            self.det_starboard_scaled_img_publisher.publish(scaled_image_msg)
            
    def stern(self, msg: CompressedImage) -> None:
        """
        Handles the stern processing of the received message.
        
        Arguments:
            msg (CompressedImage): The received image message.
        """
        
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        bboxes_pub, res_plot = self.image_callback(cv_image)
        self.bbox_stern_publisher.publish(bboxes_pub)
        
        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # No publishing options enabled, do nothing
            pass

        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_stern_scaled_img_publisher.publish(scaled_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_stern_comp_img_publisher.publish(compressed_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish compressed and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_stern_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw image
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_stern_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_stern_scaled_img_publisher.publish(scaled_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw and compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_stern_comp_img_publisher.publish(compressed_image_msg)
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_stern_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw, compressed, and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_stern_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)
            self.det_stern_scaled_img_publisher.publish(scaled_image_msg)
        
    def portside(self, msg: CompressedImage) -> None:
        """
        Handles the portside processing of the received message.
        
        Arguments:
            msg (CompressedImage): The received image message.
        """
        
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        bboxes_pub, res_plot = self.image_callback(cv_image)
        self.bbox_portside_publisher.publish(bboxes_pub)
        
        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # No publishing options enabled, do nothing
            pass

        if (not self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_portside_scaled_img_publisher.publish(scaled_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_portside_comp_img_publisher.publish(compressed_image_msg)

        if (not self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish compressed and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_portside_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw image
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_portside_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (not self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            self.det_portside_scaled_img_publisher.publish(scaled_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (not self.publish_scaled):
            # Publish raw and compressed image
            compressed_image_msg = self.create_compressed_img(res_plot)
            self.det_portside_comp_img_publisher.publish(compressed_image_msg)
            raw_image_msg = self.bridge.cv2_to_imgmsg(res_plot, encoding='bgr8')
            self.det_portside_img_publisher.publish(raw_image_msg)

        if (self.raw_img_publish) and (self.compressed_img_publish) and (self.publish_scaled):
            # Publish raw, compressed, and scaled image
            scaled_image_msg = self.create_scaled_img(res_plot)
            scaled_image = self.bridge.imgmsg_to_cv2(scaled_image_msg)
            compressed_scaled_image_msg = self.create_compressed_img(scaled_image)
            self.det_portside_comp_scaled_img_publisher.publish(compressed_scaled_image_msg)
            self.det_portside_scaled_img_publisher.publish(scaled_image_msg)
            
            
    def create_compressed_img(self, orig_img: np.ndarray) -> None:
        """
        Creates a compressed image message from the original image.

        Args:
            orig_img (numpy.ndarray): The original image.

        Returns:
            CompressedImage: The compressed image message.
        """
        compressed_frame = self.jpeg.encode(orig_img, quality=self.compression_quality,
                                                flags=(TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT))
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        compressed_image_msg = CompressedImage()
        compressed_image_msg.format = "jpeg"
        compressed_image_msg.header = h
        compressed_image_msg.data = compressed_frame
        return compressed_image_msg
    
    def create_scaled_img(self, orig_img: np.ndarray) -> None:
        """
        Creates a scaled image message from the original image.
        
        Arguments:
            orig_img (numpy.ndarray): The original image.
        
        Returns:
            Image: The scaled image message.
        """
        scaled_img = cv2.resize(orig_img,(self.width,self.height), interpolation=cv2.INTER_AREA)
        image_ros_cv = self.bridge.cv2_to_imgmsg(scaled_img, encoding='bgr8')
        return image_ros_cv
                        

def main(args=None) -> None:
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)
    
    # Create a ROS2 node with the name 'yolo_detector'
    node = rclpy.create_node('yolo_detector')
    
    # Create an instance of the YoloDetectorNode class
    yolo_detector = YoloDetector(node)
    
    # Enter the ROS2 event loop
    rclpy.spin(node)
    
    # Print a shutdown message
    node.get_logger().info('Shutting down')
    
    # Clean up and destroy the YoloDetectorNode
    node.destroy_node()
    
    # Shutdown the ROS2 Python client library
    rclpy.shutdown()
    
    exit(0)


if __name__ == '__main__':
    main()
