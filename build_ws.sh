#!/usr/bin/bash

# Make sure the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: Please source this script instead of running it."
    exit 1
fi

# Setup ROS2 environment
source /opt/ros/foxy/setup.bash

echo "Building using Colcon"
colcon build --symlink-install

source install/setup.bash
