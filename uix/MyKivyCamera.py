from kivy.uix.image import Image
import kivy
from kivy.clock import Clock
from uix.kivy_useful_func import convert_opencv_to_texture
import cv2
from threading import Lock
from threading import Thread


class MyKivyCamera:
    def __init__(self, fps=30, auto_start=False):
        self._android = kivy.platform == 'android'
        self._frame_rate = fps
        self._last_image_read = None

        self._capture = self._init_capture()
        self._paused = False
        self._should_stop = False
        self._lock = Lock()
        self._thread = None

        if auto_start:
            self.start()

    def start(self):
        self._thread = Thread(target=self._update_loop, name="Thread Video Capture", args=())
        self._thread.daemon = True
        self._thread.start()
        return self

    def _init_capture(self):
        if self._android:
            capture = None
        else:
            capture = cv2.VideoCapture(-1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, self._frame_rate)

        return capture

    def _update_loop(self):
        while not self._should_stop:
            if self._paused:
                self.retrieve_img()
            else:
                ret, frame = self.read_img()
                if ret:
                    with self._lock:
                        self._last_image_read = frame

    @property
    def last_image_read(self):
        return self._last_image_read

    def stop_capture(self):
        if self._android:
            pass
        else:
            self._capture.release()
        self._capture = None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def read_img(self):
        if self._capture is None:
            return False, None

        if self._android:
            return False, None

        else:
            return self._capture.read()

    def retrieve_img(self):
        if self._capture is None:
            pass

        elif self._android:
            pass

        else:
            self._capture.retrieve()
