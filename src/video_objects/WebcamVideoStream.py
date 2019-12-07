import sys

import cv2

from settings import CAMERA_PORT
from src.useful_functions import show_and_check_close, find_available_camera_port
from src.video_objects.VideoStream import *


class WebcamVideoStream(VideoStream):
    def __init__(self, src=CAMERA_PORT,
                 name="WebcamVideoStream", display=False,
                 nb_threads=None):
        VideoStream.__init__(self, name, display=display)
        self.stream = cv2.VideoCapture(src)
        if self.stream is None or not self.stream.isOpened():
            list_cameras = find_available_camera_port()
            list_cameras_str = " - ".join([str(w) for w in list_cameras])
            logger.error(20 * "-")
            logger.error('Unable to open video source: ' + str(src))
            logger.error("Available cam ports are : " + list_cameras_str)
            sys.exit(4)

        (self.grabbed, self.frame) = self.stream.read()
        logger.debug("Camera Object created and launched")
        if nb_threads is not None:
            cv2.setNumThreads(nb_threads)
        logger.info("OpenCV is using {} threads".format(cv2.getNumThreads()))

    def update(self):
        # cv2.namedWindow(self.name)
        # cv2.createTrackbar("quality", self.name, 90, 100, nothing)

        while not self.should_stop:
            (self.grabbed, frame_obtain) = self.stream.read()
            if not self.grabbed:
                logger.warning("Video Capture Read Nothing")
                continue
            self.get_new_timestamp()
            with self.lock:
                self.frame = frame_obtain.copy()
            if self.display:
                if show_and_check_close(self.name, self.frame):
                    self.should_stop = True
        self.stop()

    def stop(self):
        logger.debug("Stopping Webcam's Stream")

        # del self.stream
        self.stopped = True

    def adjust_stream_settings(self):
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def get(self, *args):
        return self.stream.get(*args)

    def release(self):
        del self
