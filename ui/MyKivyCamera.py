from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import kivy
from kivy.clock import Clock

import cv2


# noinspection PyArgumentList
def convert_opencv_to_texture(frame):
    if frame is None:
        return Texture()
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tostring()
    image_texture = Texture.create(
        size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    return image_texture


def convert_texture_to_opencv(texture):
    pixels = texture.pixels
    img_opencv = cv2.flip(pixels, 0)

    return img_opencv


class MyKivyCamera(Image):

    def __init__(self, fps=30, auto_start=False, **kwargs):
        super(Image, self).__init__(**kwargs)

        self.android = kivy.platform == 'android'
        self.__freeze = False
        self.last_image_read = None

        self.capture = None
        Clock.schedule_interval(self.update, 1.0 / fps)

        if auto_start:
            self.start_capture()

    def update(self, _dt):
        ret, frame = self.read_img()
        if ret:
            self.last_image_read = frame

            if not self.__freeze:
                # display image from the texture
                self.texture = convert_opencv_to_texture(frame)

    def start_capture(self):
        if self.android:
            self.capture = None
        else:
            self.capture = cv2.VideoCapture(-1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def stop_capture(self):
        if self.android:
            pass
        else:
            self.capture.release()
        self.capture = None

    def read_img(self):
        if self.capture is None:
            return False, None

        if self.android:
            return False, None

        else:
            return self.capture.read()

    def freeze(self):
        self.__freeze = True

    def unfreeze(self):
        self.__freeze = False
