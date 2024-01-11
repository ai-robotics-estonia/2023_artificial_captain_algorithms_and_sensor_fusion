# Nvidia L4T base image
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ARG DEBIAN_FRONTEND=noninteractive

# Link trtexec to /usr/bin
RUN ln -s /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec

# We need upgraded system packages for ROS2
RUN apt update
RUN apt install -y apt-transport-https \
                   ca-certificates \
                   curl \
                   gnupg-agent \
                   software-properties-common

# OpenCV in this container is messed up, so we need to purge it
# We will install new one later
RUN apt purge -y '*opencv*'

# Upgrade system packages
RUN apt upgrade -y

# Upgrade pip
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

# ROS2 needs locales to be set up properly
RUN apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# ROS2 needs the universe repository
RUN add-apt-repository universe

# Install common dependencies
RUN apt install -y build-essential \
                   cmake \
                   git \
                   wget \
                   unzip \
                   gnupg2 \
                   lsb-release \
                   python-is-python3

# Add ROS2 apt key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 apt repository
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update

# Install ROS2 Foxy
RUN apt install -y ros-foxy-ros-base \
                   python3-argcomplete \
                   python3-colcon-common-extensions

# Install ROS2 development tools
RUN apt install -y ros-dev-tools

# Install Mosquitto MQTT clients
RUN apt install -y mosquitto-clients

# Install workspace dependencies from APT
ADD ./scripts/install/install_depts_apt.sh /install/install_depts_apt.sh
RUN bash /install/install_depts_apt.sh

# Install workspace dependencies using rosdep
RUN . /opt/ros/foxy/setup.sh

# Copy source code into container so we can use rosdep to install dependencies
RUN mkdir -p /asv_tmp/src
ADD ./src /asv_tmp/src

RUN rosdep init
RUN rosdep update -include-eol-distros
RUN rosdep install -y --from-paths /asv_tmp/src --ignore-src --rosdistro foxy -r --skip-keys "console_bridge fastcdr fastrtps rti-connext-dds-5.3.1 urdfdom_headers"

RUN rm -rf /asv_tmp/src

# Cleanup
RUN apt autoremove -y
RUN apt clean

RUN pip3 cache purge

# Remove /install directory
RUN rm -rf /install

# Set timezone to Europe/Tallinn
RUN ln -fs /usr/share/zoneinfo/Europe/Tallinn /etc/localtime

ARG USERNAME=aju

# Switch to non-root user
RUN useradd -m -s /bin/bash $USERNAME
RUN usermod -aG video $USERNAME
USER $USERNAME

# Install workspace dependencies from pip
ADD ./scripts/install/requirements.txt /install/requirements.txt
RUN pip3 install --upgrade -r /install/requirements.txt

# Since we had to purge OpenCV, we need to install TensorFlow and ONNX again...
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
RUN pip3 install git+https://github.com/onnx/tensorflow-onnx@v1.14.0

# Downgrade setuptools to 58.2.0 so we don't get deprecation warnings
RUN pip3 install setuptools==58.2.0

RUN pip3 cache purge

# Source ROS2 environment
RUN echo "source /opt/ros/foxy/setup.bash" >> /home/$USERNAME/.bashrc

# If workspace is built, source it
RUN echo "if [ -f /asv-ros/install/setup.bash ]; then source /asv-ros/install/setup.bash; fi" >> /home/$USERNAME/.bashrc

# Replace the PS1 prompt with something more useful
RUN echo "export PS1='(asv-ros)\[\033[01;33m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /home/$USERNAME/.bashrc

# Change working directory
WORKDIR /asv-ros
