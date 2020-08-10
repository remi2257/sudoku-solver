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


class KivyCamera(Image):

    def __init__(self, fps=30, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(-1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.last_image_read = None

    def update(self, _dt):
        ret, frame = self.capture.read()
        if ret:
            self.last_image_read = frame

            # display image from the texture
            self.texture = self.convert_to_texture(frame)

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()

    # noinspection PyArgumentList
    @classmethod
    def convert_to_texture(cls, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        return image_texture


class SudokuCameraWindow(Screen):
    camera = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(SudokuCameraWindow, self).__init__(**kwargs)

        self.model = load_model(model_default_name)

    def solve(self):
        frame = self.camera.last_image_read
        worked_frame = process_single_img(frame, self.model)
        self.apply_texture_from_frame(worked_frame)

    def apply_texture_from_frame(self, frame):
        self.manager.ids.solver.image.texture = KivyCamera.convert_to_texture(frame)
        self.manager.transition.direction = "left"
        self.manager.current = "solver"


class SolverWindow(Screen):
    image = ObjectProperty(None)

    def get_back_to_camera(self):
        self.manager.transition.direction = "right"
        self.manager.current = "main"


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("sudoku.kv")


class SudokuSolverApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    SudokuSolverApp().run()
