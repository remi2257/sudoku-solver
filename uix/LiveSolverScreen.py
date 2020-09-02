from kivy.clock import Clock
from kivy.uix.image import Image
from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivymd.uix.button import MDFillRoundFlatIconButton

from uix.MyKivyCamera import MyKivyCamera

from uix.pimped_widgets import SwitchWithText
from uix.kivy_useful_func import convert_opencv_to_texture

from tensorflow.keras.models import load_model
from src.main_single_img import process_single_img

model_default_name = 'model/my_model.h5'


class ButtonSolve(MDFillRoundFlatIconButton):
    dict_param = {
        "solve": {
            "text": "Solve",
            "icon": "yoga",
        },
        "unfreeze": {
            "text": "Unfreeze",
            "icon": "air-horn",
        },
    }

    def __init__(self, cb, **kwargs):
        super().__init__(**self.dict_param["solve"], **kwargs)
        self.__is_freeze = False
        self.cb = cb

    @property
    def is_freeze(self):
        return self.__is_freeze

    def on_release(self):
        super().on_release()
        param_name = "solve" if self.__is_freeze else "unfreeze"
        params = self.dict_param[param_name]
        self.text = params["text"]
        self.icon = params["icon"]

        self.__is_freeze = not self.__is_freeze

        self.cb(self.__is_freeze)


class LiveSolverScreen(MDBottomNavigationItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Live Solver'
        self.icon = 'camera'

        # -- UI
        self.big_relative_layout = RelativeLayout()

        # - Video Stream
        self.video_stream = MyKivyCamera(auto_start=True)
        self.image = Image(size_hint_y="0.75", pos_hint={'top': 0.95}, id="image")
        self.fps = 30
        self.last_image_read = None

        Clock.schedule_interval(self.read_video_stream, 1.0 / self.fps)

        # - Option Layout
        self.layout_options = MDBoxLayout(size_hint_y="0.15", pos_hint={'y': 0.05}, id="options")

        self.hint_mode_switch = SwitchWithText(text="Hint Mode",
                                               size_hint_x=0.15,
                                               id_="quality", active=True)

        self.solve_unfreeze_button = ButtonSolve(cb=self.callback_solve_unfreeze_button)

        self.layout_options.add_widget(self.hint_mode_switch)
        self.layout_options.add_widget(self.solve_unfreeze_button)

        self.layout_options.ids = {child.id: child for child in self.layout_options.children}

        self.big_relative_layout.add_widget(self.image)
        self.big_relative_layout.add_widget(self.layout_options)

        self.add_widget(self.big_relative_layout)

        # - Solver
        self.model = load_model(model_default_name)

    def read_video_stream(self, _dt):
        if self.parent is None:
            self.video_stream.pause()
            return
        else:
            self.video_stream.resume()

        frame = self.video_stream.last_image_read

        self.last_image_read = frame
        if frame is None:
            return

        if not self.solve_unfreeze_button.is_freeze:
            self.set_new_image()

    def set_new_image(self):
        self.image.texture = convert_opencv_to_texture(self.last_image_read)

    def should_give_only_hint(self):
        return self.hint_mode_switch.is_active()

    def callback_solve_unfreeze_button(self, is_freeze):
        # is_freeze = self.solve_unfreeze_button.is_freeze
        print("Is freeze ??", is_freeze)

        if is_freeze:
            self.solve()

    def solve(self):
        hint_mode = self.should_give_only_hint()
        solved_frame = process_single_img(self.last_image_read, self.model, hint_mode=hint_mode)
        self.display_solved(solved_frame)

    def display_solved(self, solved_frame):
        self.last_image_read = solved_frame
        self.set_new_image()
