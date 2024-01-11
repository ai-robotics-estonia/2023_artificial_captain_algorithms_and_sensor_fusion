# Ship Detection on the Sea using YOLO and ROS2

This repository provides an implementation of ship detection on the sea using the YOLO (You Only Look Once) algorithm and ROS2 (Robot Operating System 2). The project uses a custom ROS message format.

## Installation

To use this project, follow the steps below:

1. Clone the repository:

   ```
   git clone https://github.com/canersu/ship_detector.git
   ```
2. Install the required dependencies. Make sure you have ROS2 installed and properly configured.
3. Build the project:

   ```
   cd 'path to ROS2 workspace'&& colcon build
   ```
4. Source the ROS2 setup file:

   ```
   source install/setup.bash
   ```

## Usage

1. Launch the ROS2 node responsible for ship detection:

   ```
   ros2 launch ship_detection detector_node_launch.py
   ```
2. The node will start subscribing to the appropriate ROS topics, where input sensor data (e.g., images) are published.
3. The YOLO-based ship detection algorithm will process the input data and publish the detected ships as custom ROS messages.
4. You can subscribe to the detection results by accessing the corresponding ROS topic.

## Custom ROS Message Format

The custom ROS message format used in this project includes the following fields:

* `float64 probability`: Represents the probability score of the detected ship. The higher the probability, the more confident the detection.
* `int64 xmin`: Represents the minimum x-coordinate of the bounding box surrounding the ship.
* `int64 ymin`: Represents the minimum y-coordinate of the bounding box surrounding the ship.
* `int64 xmax`: Represents the maximum x-coordinate of the bounding box surrounding the ship.
* `int64 ymax`: Represents the maximum y-coordinate of the bounding box surrounding the ship.
* `int16 id`: Represents a unique identifier for the detected ship.
* `int16 class_id`: Represents the class identifier of the detected ship. This field is used when multiple classes of objects are detected.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://chat.openai.com/LICENSE). You are free to use, modify, and distribute the code in accordance with the terms specified in the license.

## Acknowledgments

This project is built upon the YOLO algorithm and ROS2 framework. We would like to acknowledge the creators and contributors of YOLO and ROS2 for their valuable work.

## Contact

For any questions or inquiries, please contact:

Your Name
Email: [canersu34@gmail.com](mailto:yourname@example.com)
