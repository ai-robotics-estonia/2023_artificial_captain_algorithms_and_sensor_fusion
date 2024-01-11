import numpy as np

from queue import Queue
from threading import Lock

import rclpy
import rclpy.qos as qos

from builtin_interfaces.msg import Time

class ThreadInterface:
    def __init__(self, thread_id: int, topic: str) -> None:
        self._id = thread_id
        self._lock = Lock()
        self._queue = Queue(maxsize=1)
        self._topic = topic
        
    def get_id(self) -> int:
        return self._id
        
    def get_lock(self) -> Lock:
        return self._lock
    
    def get_queue(self) -> Queue:
        return self._queue
    
    def get_topic(self) -> str:
        return self._topic
    
class Frame:
    def __init__(self, image: np.ndarray, timestamp: Time, frame_id: str) -> None:
        self._image = image
        self._timestamp = timestamp
        self._frame_id = frame_id
        
    def get_image(self) -> np.ndarray:
        return self._image
    
    def get_timestamp(self) -> Time:
        return self._timestamp
    
    def get_timestmp_as_sec(self) -> float:
        return self._timestamp.sec
    
    def get_frame_id(self) -> str:
        return self._frame_id

def create_sensor_qos_profile(buffer_depth: int, lease_duration: int) -> qos.QoSProfile:
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