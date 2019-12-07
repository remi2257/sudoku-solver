"""
Mother class for VideoStream classes
"""

import time
from threading import Lock
from threading import Thread

from numpy import mean

from settings import logger
from src.useful_functions import add_annotation_to_image


class VideoStream:
    # calibration_obj: CalibrationObject

    def __init__(self, name, display=False):
        logger.debug("Creating Video Stream Object")
        self.name = name
        self.display = display
        self.frame = None

        self.lock = Lock()
        self.thread = None

        self.stopped = False
        self.should_stop = False

        self.timestamp_last_frame = time.time()

        self.last_read_timestamp = self.timestamp_last_frame
        self.fps_list = [0.0 for _ in range(30)]

    # def start(self):
    #     raise NotImplementedError
    def start(self):
        logger.debug("Creating VideoStream's thread")
        self.thread = Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        raise NotImplementedError

    def get_new_timestamp(self):
        self.timestamp_last_frame = time.time()

    def read(self):
        with self.lock:
            return self.frame.copy()
            # return True, self.frame.copy()

    def read_timestamp(self):
        return self.timestamp_last_frame

    def get_img_and_timestamp(self):
        with self.lock:
            return self.frame.copy(), self.timestamp_last_frame

    def read_w_timestamp(self):
        new_read_timestamp = time.time()
        img = self.read()

        self.fps_list.pop(0)

        measured_framerate = 1 / (new_read_timestamp - self.last_read_timestamp)
        self.fps_list.append(measured_framerate)
        img = add_annotation_to_image(img, "{:.1f} FPS".format(mean(self.fps_list)), write_on_top=True)

        self.last_read_timestamp = new_read_timestamp

        return img

    def stop(self):
        raise NotImplementedError

    def __del__(self):
        if self.thread is not None:
            with self.lock:
                self.thread.join()
