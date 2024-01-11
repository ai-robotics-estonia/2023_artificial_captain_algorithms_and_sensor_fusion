#!/usr/bin/env python3
from detection_interface.msg import BoundingBox, BoundingBoxes
from detection_interface.msg import BearingAngle, BearingAngles
from sensor_msgs.msg import CameraInfo
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
import math

class BearingFinder(Node):
    def __init__(self):
        # Initialize ROS 2 node
        super().__init__('bearing_finder')
        self.subscriber = self.create_subscription(BoundingBoxes, '/frontpi/detections/box_coordinates', self.bounding_boxes_callback, 5)
        self.subscriber = self.create_subscription(CameraInfo, '/frontpi/oak/rgb/camera_info', self.camera_info_callback, 5)
        #self.pub_bearings = self.create_publisher(Float64MultiArray, '/frontpi/detections/bearingAngles', 5)
        self.pub_bearing_angles = self.create_publisher(BearingAngles, '/frontpi/detections/bearingAngles', 5)
        
    def calculateBoudingBox_centre(self, box_coords):
        box_centre_coord_pixel = [box_coords.xmin + (int)((box_coords.xmax - box_coords.xmin)/2), 
                               box_coords.ymin + (int)((box_coords.ymax - box_coords.ymin)/2)]          
        #box_centre_coord = [(np.float32)(box_centre_coord_pixel[0]/self.image_width), (np.float32)(box_centre_coord_pixel[1]/self.image_height)]
        return box_centre_coord_pixel
    
    def findPointWorld_cvFunction(self, point_coord_pixel):
        camera_matrix = np.array(self.K_matrix.reshape(3,3), dtype=np.float32)
        dist_coeffs = np.array(self.distortion_coeffs, dtype=np.float32) 
        #print("point_coord", point_coord_pixel)
        xy_undistorted = cv2.undistortPoints(point_coord_pixel, camera_matrix, dist_coeffs)
        dst = np.squeeze(xy_undistorted)
        return dst    
    
    def findPointWorld_inverseMatrix(self,point_coord_pixel):
        #projection of pixel points to real world coordinates
         #matrix inverse
        camera_matrix = np.array(self.K_matrix.reshape(3,3), dtype=np.float32)
        K_matrix_inv=np.linalg.inv(camera_matrix)
        w=point_coord_pixel[0]
        h=point_coord_pixel[1]
        r = K_matrix_inv.dot([w, h, 1.0])
        #returs ray in world space of camera [float, float, float]
        return r
        
    def calculateBearingAngle(self, ray):
        angle_radians = np.arccos(ray[0])
        print("angle radians", angle_radians)
        angle_degrees = math.degrees(angle_radians)
        bearing_angle=angle_degrees
        print("angle degrees", angle_degrees)
        print("ray", ray)
        if(ray[0]<0):
            angle_degrees=angle_degrees-90
            bearing_angle=360-angle_degrees
        elif(ray[0]>=0):
            angle_degrees=90-angle_degrees
            bearing_angle=angle_degrees
        print("angle degrees after normalization", angle_degrees)
        return bearing_angle
        
    
    def bounding_boxes_callback(self, box_coords):
        det_box_coords = box_coords.bounding_boxes
        ba_object = BearingAngle()
        bearing_angles = []
        bearing_angles_msg = BearingAngles()
        print(det_box_coords)
        for i in range(len(det_box_coords)):
            box_centre_coord=self.calculateBoudingBox_centre(det_box_coords[i])
            print("box_centre coord: ", box_centre_coord[0], box_centre_coord[1])
            box_centre_coord = np.array(box_centre_coord, dtype=np.float32)
            #method 1       
            r_dst=self.findPointWorld_cvFunction(box_centre_coord)
            print("function r_dst", r_dst)
            #method 2
            r_inv=self.findPointWorld_inverseMatrix(box_centre_coord)
            print("inverse matrix r_inv", r_inv)
            
            bearing_angle_inv= self.calculateBearingAngle(r_inv)
            print("bearing angle inv matrix", bearing_angle_inv)
            
            bearing_angle_dst= self.calculateBearingAngle(r_dst)
            print("bearing angle function", bearing_angle_dst)
            ba_object.angle=bearing_angle_dst
            ba_object.id= det_box_coords[i].id
            bearing_angles.append(ba_object)
        bearing_angles_msg.bearing_angles = bearing_angles
        self.pub_bearing_angles.publish(bearing_angles_msg)
            
           
    def camera_info_callback(self, msg):
        self.image_height=msg.height
        self.image_width=msg.width
        self.K_matrix= msg.k
        self.distortion_coeffs=msg.d
        
        #pub_lists = Float64MultiArray()
        #pub_lists.data = [1.0,2.1,3.2]      
        #self.pub_bearings.publish(pub_lists)
        print("height and width of frame", self.image_height, self.image_width)
        

def main(args=None) -> None:
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)
    
    # Create an instance of the BearingFinderrNode class
    bearing_finder = BearingFinder()
    
    # Enter the ROS2 event loop
    rclpy.spin(bearing_finder)
    
    # Print a shutdown message
    print("shutdown")
    
    # Clean up and destroy the BearingFinderNode
    bearing_finder.destroy_node()
    
    # Shutdown the ROS2 Python client library
    rclpy.shutdown()

    exit(0)

if __name__ == '__main__':
    main()
