#!/bin/bash

# Define the name of your Docker Compose service
SERVICE_NAME="asv_ros"

# Check if .bash_history exists in the current directory
# If it does not exist, create it. If it is a directory,
# Delete it and create a file instead.
if [ -e .bash_history ]; then
  if [ -d .bash_history ]; then
    rm -rf .bash_history
    touch .bash_history
  fi
else
  touch .bash_history
fi

echo "Starting Docker Compose service '$SERVICE_NAME'..."
# Check if there is a running container with the specified name
CONTAINER_ID=$(docker ps -q -f name="$SERVICE_NAME")
echo "CONTAINER_ID: $CONTAINER_ID"

if [ -z "$CONTAINER_ID" ]; then
  echo "Container with name '$SERVICE_NAME' not found."
  # Container is not running, so start it using Docker Compose
  INTERACTIVE=true docker compose up -d
  # Wait for a moment to ensure the container is started
  sleep 2
  # Get the ID of the first running container with the specified name
  CONTAINER_ID=$(docker ps -q -f name="$SERVICE_NAME" | head -n 1)
fi

echo "CONTAINER_ID: $CONTAINER_ID"

# Enter the container using docker exec
if [ -n "$CONTAINER_ID" ]; then
  echo "Entering container with ID '$CONTAINER_ID'..."
  docker exec -it "$CONTAINER_ID" /bin/bash
else
  echo "Container with name '$SERVICE_NAME' not found or unable to start."
  exit 1
fi
