import numpy as np
from builtin_interfaces.msg import Time

class Frame:
    def __init__(self, image: np.ndarray, timestamp: Time) -> None:
        self.image = image
        self.timestamp = timestamp

    def get_image(self) -> np.ndarray:
        return self.image
    
    def get_timestamp(self) -> Time:
        return self.timestamp
    
    def get_timestmp_as_sec(self) -> float:
        return self.timestamp.sec
    