import os

import numpy as np
from time import time

from ultralytics import YOLO

from typing import List

import rclpy
from rclpy.node import Node

from src.common import ThreadInterface, Frame
from src.deepsort_tracker.deepsort import DeepsortWrapper


class ObjectDetection(Node):
    def __init__(self, sub_interfaces: List[ThreadInterface], pub_interfaces: List[ThreadInterface], publish_images: bool) -> None:
        """
        ROS2 node for object detection. It receives images from a ROS2 topic, performs object detection,
        and publishes the bounding boxes to another ROS2 topic. Optionally, it can also publish the images
        with the bounding boxes drawn on them.

        Args:
            sub_interfaces (List[ThreadInterface]): List of subscriber thread interfaces
            pub_interfaces (List[ThreadInterface]): List of publisher thread interfaces
            publish_images (bool): Whether to publish images or not
            
        Returns:
            None
        """
        
        # Call the parent constructor
        super().__init__('ML_detector_object_detection')
        
        self.logger = self.get_logger()
        
        self.logger.info(f"OBJ_DET: Initializing")
        
        self.sub_interfaces = sub_interfaces
        self.pub_interfaces = pub_interfaces
        
        self.publish_images = publish_images
        
        # Declare parameters
        
        # Debugging
        self.declare_parameter('debug', 'False')
        
        # YOLO config
        self.declare_parameter('yolo_model', '')
        self.declare_parameter('input_size', 0)
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('iou_threshold', 0.3)
        
        # Tracker config
        self.declare_parameter('enable_tracker', 'False')
        self.declare_parameter('tracker_type', 'bytetrack')
        
        # Read parameters
        
        # Debugging
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # YOLO config
        self.yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_thresh = self.get_parameter('iou_threshold').get_parameter_value().double_value
        
        # Tracker config
        self.enable_tracker = self.get_parameter('enable_tracker').get_parameter_value().bool_value
        
        if self.yolo_model == '':
            raise Exception('YOLO model not specified!')
        
        if self.input_size == 0:
            raise Exception('Model input size not specified!')
        
        # Model initialization
        self.classes = [0, 1]
        
        # Load the model
        self.logger.info(f"OBJ_DET: Loading model")
        self.model = YOLO(self.yolo_model, task='detect')
        self.logger.info(f"OBJ_DET: Model loaded")
       
        if self.enable_tracker:
            self.logger.info(f"OBJ_DET: Initializing tracker")
            
            raise RuntimeError("Tracker not fully implemented yet!")
            
            # Get the root directory of the current Python script
            root_directory = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(root_directory, 'deepsort_tracker/ckpts/model640_new_weights.pt')
            self.tracker = DeepsortWrapper(model=model_path)
            self.logger.info(f"OBJ_DET: Tracker initialized")
        
        # ROS2 hack to emulate an infinite loop      
        self.loop_timer = self.create_timer(0.03, self._process_frames) # 30FPS max
    
        self.logger.info(f"OBJ_DET: Initialization complete")
    
    def _format_objects_for_deepsort(objs, confidence, min_width):
        out_scores = []
        detections = []
        obj_classes = []

        for obj in objs:
            if obj.confidence >= confidence:

                box_w = obj.bounding_box.xmax - obj.bounding_box.xmin

                if box_w >= min_width:
                    box_h = obj.bounding_box.ymax - obj.bounding_box.ymin

                    detected_bbox = [obj.bounding_box.xmin,
                                    obj.bounding_box.ymin,
                                    box_w,
                                    box_h]

                    detections.append(detected_bbox)
                    out_scores.append(obj.confidence)
                    obj_classes.append(obj.obj_class)

        return np.array(out_scores), np.asarray(detections), np.asarray(obj_classes)
    
    def _process_frames(self) -> None:
        """
        Receive frames from the subscriber and perform object detection on them.
        Then, publish the bounding boxes.
        Essentially a mux-ML-demux solution. One model, multiple cameras
        
        Args:
            None
            
        Returns:
            None
        """
        
        for i, sub_interface in enumerate(self.sub_interfaces):
            
            # Read the frame from the queue
            if not sub_interface.get_queue().empty() and rclpy.ok():
                frame = None
                with sub_interface.get_lock():
                    frame = sub_interface.get_queue().get()
                if frame is None:
                    continue
                img = frame.get_image()
                    
                # Perform object detection
                classes, box_conf, box_coords, num_detections, elapsed_time, img_det = self.detect_objects(img)
                if self.enable_tracker:
                    tracker, detections_class = self.tracker.run_deep_sort(img, box_conf, box_coords, classes)
                    
                if self.publish_images:
                    pub_interface = self.pub_interfaces[i]
                    if not pub_interface.get_queue().full():
                        # Publish the bounding boxes
                        send_frame = Frame(img_det, frame.get_timestamp(), frame.get_frame_id())
                        with pub_interface.get_lock():
                            pub_interface.get_queue().put(send_frame)
                        
        self.loop_timer.reset()
                
    def detect_objects(self, img, normalized=False):
    
        start_time = time()
        classes = np.zeros(shape=(0,1))
        box_conf = np.zeros(shape=(0,1))
        box_coords = np.zeros(shape=(0,4))
        num_detections = 0
        
        # results = self.model.track(img,
        #                            conf=self.conf_thresh, 
        #                            half=True,
        #                            classes=self.classes,
        #                            device=0,
        #                            iou=self.iou_thresh,
        #                            imgsz=self.input_size,
        #                            tracker=self.tracker_type)

        results = self.model(img,
                             conf=self.conf_thresh, 
                             half=False,
                             classes=self.classes,
                             device=0,
                             iou=self.iou_thresh,
                             imgsz=self.input_size,
                             verbose=False)
        boxes = results[0].boxes
        num_detections = len(results[0])
        
        if self.debug:
            self.logger.info(f"OBJ_DET: ******** Detected {num_detections} objects ********")
        
        if normalized:
            bbox_coords = boxes.xyxyn.to('cpu').numpy()
        else:
            bbox_coords = boxes.xyxy.to('cpu').numpy()
        
        if self.debug:
            self.logger.info(f"OBJ_DET: ******** Bounding box coordinates: {bbox_coords} ********")
        
        end_time = time()

        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 3)
        
        if self.debug:
            self.logger.info(f"OBJ_DET: ******** Elapsed time: {elapsed_time} seconds ********")
        
        return boxes.cls.to('cpu').numpy(), boxes.conf.to('cpu').numpy(), bbox_coords, num_detections, elapsed_time, results[0].plot()



# if __name__ == '__main__':
#     tracker = "bytetrack.yaml"
#     yolo_model = "/media/agx/yolo_weights/rt_models/fp16/640/yolov8n.engine"
#     input_size = 640
#     conf_thresh = 0.3
#     iou_thresh = 0.3
#     classes = [8]
#     OD = ObjectDetection(yolo_model, input_size, conf_thresh, iou_thresh, tracker, classes)