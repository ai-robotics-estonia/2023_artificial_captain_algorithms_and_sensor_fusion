#!/bin/bash

cd /asv-ros

source /opt/ros/foxy/setup.bash
source install/setup.bash

echo "Interactive: $INTERACTIVE"

# Check if the INTERACTIVE environment variable is set and not empty
if [ -n "$INTERACTIVE" ] && [ "$INTERACTIVE" = "true" ]; then
    # This is not an entrypoint
    echo "Container is started for interactive mode."
    # Infinite loop to keep the container running. Quit with CTRL+C
    while true; do sleep 1; done

else
    # This is being run as an entrypoint, execute ROS2 launch
    # Read the launch file from the ASV_LAUNCH_FILE environment variable
    if [ -z "$ASV_LAUNCH_FILE" ]; then
        echo "ASV_LAUNCH_FILE environment variable is not set. Exiting."
        exit 1
    fi
    echo "Launch file: $ASV_LAUNCH_FILE"
    ros2 launch $ASV_LAUNCH_FILE
fi
