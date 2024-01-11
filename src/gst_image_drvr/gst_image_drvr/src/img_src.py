import gi
import cv2
import time
import numpy as np
from queue import Queue
from threading import Thread, Lock, Event

from rclpy.node import Node

from src.common import Frame

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GstApp

APPSINK_STR = 'appsink name=appsink max-buffers=1 drop=true sync=false'

class CameraSrc(Thread):
    def __init__(self, node: Node, mutex: Lock, queue: Queue, img_event: Event) -> None:
        """
        Constructor for the CameraDrv class.
        
        Args:
            node: ROS2 node object
            mutex: Lock object for thread-safe access to the queue
            queue: Queue object for storing frames
            img_event: Event object for signaling that a new frame is available
        
        Returns:
            None
        """
        
        # Call the superclass constructor
        super().__init__()
        
        # ROS2 node object
        self.node = node
        self.logger = node.get_logger()
        
        # Read parameters
        # Debug mode
        self.debug = self.node.get_parameter('debug').get_parameter_value().bool_value
        
        # Stability
        self.frame_timeout_sec = self.node.get_parameter('no_frame_timeout_sec').get_parameter_value().double_value
        
        # Publish rate
        rate = self.node.get_parameter('rate').get_parameter_value().double_value
        
        # Gstreamer pipeline
        self.pipeline_str = self.node.get_parameter('pipeline').get_parameter_value().string_value
        
        self.logger.info(f"PROD: Pipeline: {self.pipeline_str}")
            
        if self.pipeline_str == '':
            self.logger.error("PROD: Pipeline is empty!")
            raise ValueError("PROD: Pipeline is empty!")
        
        self.pipeline = None
        
        # For storing frames
        self.queue = queue
        self.mutex = mutex
        self.img_event = img_event
        
        # Program flow control
        self.should_quit = False
        self.gst_set_up = False
        self.bus = None
        self.appsink = None
        
        self.frame_delay = 1.0 / rate
        self.last_pub_time = time.time()
        self.last_frame_time = time.time()
        
        # Initialize GStreamer
        Gst.init(None)
        self._setup_camera(self.pipeline_str)
        
        if self.pipeline is None:
            raise RuntimeError("PROD: GStreamer pipeline parsing failed! "
                                 "Please verify that the pipeline string is correct.")
            
    def run(self):
        self.logger.info("PROD: Starting producer thread")
        
        if self.pipeline is None:
            raise RuntimeError("PROD: GStreamer pipeline is None when starting PROD thread!")
        # Start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Message bus
        self.bus = self.pipeline.get_bus()
        if self.bus is None:
            raise RuntimeError("PROD: GStreamer bus is None!")
        
        # Main loop
        while not self.should_quit:
            
            critical_error = False
            
            # Check for messages
            message = self.bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message is not None:
                critical_error = self._parse_gst_message(message)
                
            # Frame timeout
            current_time = time.time()
            if (current_time - self.last_frame_time) > self.frame_timeout_sec:
                self.logger.error("PROD: Frame timeout!")
                critical_error = True
            
            if critical_error:
                self.logger.error("PROD: Critical error in GStreamer pipeline")
                # Try to restart the pipeline
                time.sleep(1)
                
                # Stop the pipeline
                self.pipeline.set_state(Gst.State.NULL)
                
                # Wait for 1 second
                time.sleep(1)
                
                del self.appsink
                del self.pipeline
                
                # Re-initialize the pipeline
                self.appsink = None
                self.pipeline = None
                
                self._setup_camera(self.pipeline_str)
                time.sleep(1)
                
                # Start the pipeline again
                self.pipeline.set_state(Gst.State.PLAYING)
                
                # Reset the frame timeout
                critical_error = False
                self.last_frame_time = time.time()
            
            time.sleep(0.01)
                
        # We are exiting. Stop the pipeline
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            
        self.logger.info("PROD: Camera pipeline stopped")
                
    def stop(self):
        self.should_quit = True
        self.logger.info("PROD: Stopping producer thread")

    def _setup_camera(self, pipeline_str: str) -> None:
        """
        Setup the GStreamer camera pipeline.
        
        Args:
            pipeline_str: GStreamer pipeline string
            
        Returns:
            None
        """
        
        self.logger.info("PROD: Begin camera setup")

        # Open GStreamer pipeline
        if not '!' in pipeline_str:
            raise RuntimeError("Pipeline with only a single component!?!?")

        if pipeline_str.strip()[-1] == '!':
            pipeline_str = pipeline_str.strip()[0:-1].strip()

        pipeline_last_element = pipeline_str.strip().split('!')[-1]
        pipeline_last_element_name = pipeline_last_element.strip().split(' ')[0]
        
        # If the user already put an appsink in the pipeline, then use his/her appsink
        if 'appsink' in pipeline_last_element_name:
            if 'name=' in pipeline_last_element:
                raise RuntimeError("Please remove 'name=' parameter from appsink in the pipeline")
            else:
                pipeline_str += ' name=appsink'

        # If there is some other sink, then reject the sink in pipeline and substitute it with our own
        elif 'sink' in pipeline_last_element_name:
            pipeline_str = pipeline_str[0:pipeline_str.rfind('!')].strip()
            self.logger.warn(f"Removed '{pipeline_last_element_name}' from the pipeline "
                              "An appsink was automatically added")

        # If no appsink was found, add it
        if not 'appsink' in pipeline_last_element_name:
            pipeline_str += f' ! {APPSINK_STR}'

        self.logger.info(f"PROD: Opening pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        if self.pipeline is None:
            raise RuntimeError("GStreamer pipeline parsing failed! "
                               "Please verify that the pipeline string is correct.")

        # Connect the appsink to our program
        self.appsink = self.pipeline.get_by_name('appsink')

        if self.appsink is None:
            raise RuntimeError("Creation of appsink failed!")
        
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self._new_sample_callback)

        self.node.get_logger().info("PROD: Camera setup complete")
        self.gst_set_up = True

    def _parse_gst_message(self, message: Gst.Message) -> bool:
        """
        Parse a message received from GStreamer.
        
        Args:
            message: GStreamer message
            
        Returns:
            True if a critical error was encountered, False otherwise
        """
        
        
        # Catch all critical errors. We will try to restart the pipeline
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"PROD: GST ERROR from element {message.src.get_name()}: {err}")
            self.logger.info(f"PROD: GST ERROR debug info: {debug}")
            return True

        # End of stream. We will try to restart the pipeline
        elif message.type == Gst.MessageType.EOS:
            self.logger.info(f"PROD: GST EOS received")
            return True

        # Pipeline state changed. Record the new state
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                self.logger.info(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}, pending: {pending_state.value_nick}")

        # QoS message, usually dropped frames. Can be ignored in most cases
        elif message.type == Gst.MessageType.QOS:
            pass
        #     qos_msg = message.parse_qos_stats()
        #     if self.debug:
        #         self.logger.info(f"GST QoS warning (dropped frame?)")
        #         self.logger.info(f"QoS stats: {qos_msg}")

        # Buffering message. Log it, but ignore it
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            self.logger.warn(f"GST WARN from element {message.src.get_name()}: {warn}")
            self.logger.info(f"GST WARN debug info: {debug}")

        # Info message. Only log it if debug is enabled
        elif message.type == Gst.MessageType.INFO:
            info = message.parse_info()
            if self.debug:
                self.logger.info(f"GST INFO Message: {info}")
        
        # Stream status message. Only log it if debug is enabled
        elif message.type == Gst.MessageType.STREAM_STATUS:
            status = message.parse_stream_status()
            if self.debug:
                self.logger.info(f"GST STREAM STATUS: {status}")

        # Unknown message. Only log it if debug is enabled
        # Let's hope that it's not something important
        else:
            if self.debug:
                self.logger.info(f"Unprocessed GST message received. MSG Type: {message.type}")

        return False

    def _new_sample_callback(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """
        Processes the sample received from GStreamer and publishes it.
        This function is called by GStreamer when a new sample is available

        Parameters:
            sink: GStreamer sink instance

        Return:
            Gst.FlowReturn.OK
        """

        # Get the current time
        timestamp = self.node.get_clock().now().to_msg()

        # Get a frame from appsink
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        
        unix_time = time.time()
        self.last_frame_time = unix_time
        
        # We don't care about the sample if we are not publishing
        if (unix_time - self.last_pub_time) >= self.frame_delay:
            
            if result:
                # Get sample's parameters
                caps = sample.get_caps()

                # Get image dimensions
                frmt_str = caps.get_structure(0).get_value('format')
                h = caps.get_structure(0).get_value('height')
                w = caps.get_structure(0).get_value('width')

                # Assemble an image from the buffer
                image = np.frombuffer(mapinfo.data, np.uint8)
                image = np.reshape(image, [h, w,  -1])

                # The native format is BGR.
                # However, we support converting some formats in SW.
                if frmt_str != 'BGR':
                    if frmt_str == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    elif frmt_str == 'RGBA':
                        image = image[:,:,0:3]
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    elif frmt_str == 'BGRA':
                        image = image[:,:,0:3]

                    elif frmt_str == 'YUY2':
                        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

                    else:
                        # self.callback_error = f"Unknown image format: {frmt_str}"
                        return Gst.FlowReturn.ERROR

                frame = Frame(image, timestamp)
                self.last_pub_time = time.time()
                self._publish_frame(frame)
            
        # Clear the buffer to prevent memory leak
        buf.unmap(mapinfo)
        
        return Gst.FlowReturn.OK


    def _publish_frame(self, frame: Frame) -> None:
        """
        Send a frame to publisher thread using a queue.
        
        Args:
            frame: Frame object to publish
        
        Returns:
            None
        """
        
        # Just in case
        if not self.queue.full() and not self.img_event.is_set():
            if self.debug:
                self.logger.info("PROD: Not full")

            with self.mutex:
                self.queue.put(frame)
                self.img_event.set()
