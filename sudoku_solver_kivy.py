import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock

from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.uix.image import Image
from kivy.graphics.texture import Texture

import cv2
from tensorflow.keras.models import load_model

from src.main_single_img import process_single_img

kivy.require('1.9.0')

model_default_name = 'model/my_model.h5'


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

    def __init__(self, fps=30, **kwargs):
        super().__init__(**kwargs)

        self.android = kivy.platform == 'android'
        self.__freeze = False
        self.last_image_read = None

        self.capture = None
        Clock.schedule_interval(self.update, 1.0 / fps)

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


class SolverScreen(Screen):
    def __init__(self, **kwargs):
        super(SolverScreen, self).__init__(**kwargs)

        self.model = load_model(model_default_name)

    @property
    def target_image(self):
        raise NotImplementedError

    def solve(self):
        frame = self.target_image
        solved_frame = process_single_img(frame, self.model)
        self.display_solved(solved_frame)

    def display_solved(self, opencv_frame):
        raise NotImplementedError


class LiveSolverScreen(SolverScreen):
    camera = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(LiveSolverScreen, self).__init__(**kwargs)

        self.model = load_model(model_default_name)

    @property
    def target_image(self):
        return self.camera.last_image_read

    def display_solved(self, opencv_frame):
        self.camera.texture = convert_opencv_to_texture(opencv_frame)

    def start_stream(self):
        self.camera.start_capture()

    def stop_stream(self):
        self.camera.stop_capture()


class GallerySolverScreen(SolverScreen):
    kivy_image = ObjectProperty(None)
    original_image = None

    @property
    def target_image(self):
        return self.original_image

    def display_solved(self, opencv_frame):
        self.camera.texture = convert_opencv_to_texture(opencv_frame)


class ManualSolverScreen(Screen):
    pass


class HelpScreen(Screen):
    pass


class AboutScreen(Screen):
    pass


class MainScreen(Screen):
    pass


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("sudoku.kv")


class SudokuSolverApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    SudokuSolverApp().run()

"""
Useful stuff

def display_solved(self, frame):
    self.manager.ids.solver.image.texture = MyCamera.convert_to_texture(frame)
    self.manager.transition.direction = "left"
    self.manager.current = "solver"



"""
