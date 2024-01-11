#!/bin/bash

sudo apt update

# Install dependencies from APT, exit if apt fails
echo "Installing dependencies from APT"

sudo apt install -y --no-install-recommends libturbojpeg \
                                            ros-foxy-image-proc \
                                            protobuf-compiler \
                                            python3-protobuf \
                                            ubuntu-restricted-extras

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies from APT"
    exit 1
fi

echo "Installing Gst"
sudo apt install -y libgstreamer1.0-dev \
					libgstreamer-plugins-base1.0-dev \
					libgstreamer-plugins-bad1.0-dev \
					gstreamer1.0-plugins-base \
					gstreamer1.0-plugins-good \
					gstreamer1.0-plugins-bad \
					gstreamer1.0-plugins-ugly \
					gstreamer1.0-libav \
					gstreamer1.0-doc \
					gstreamer1.0-tools \
					gstreamer1.0-x \
					gstreamer1.0-alsa \
					gstreamer1.0-gl \
					gstreamer1.0-gtk3 \
					gstreamer1.0-qt5 \
					gstreamer1.0-pulseaudio

if [ $? -ne 0 ]; then
    echo "Failed to install Gst"
    exit 1
fi

echo "Cleaning up APT cache"
sudo apt autoremove -y
sudo apt clean -y
