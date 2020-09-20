from kivy.clock import Clock
from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivymd.uix.button import MDFillRoundFlatIconButton

from uix.MyKivyCamera import MyKivyCamera

from uix.pimped_widgets import MyToggleButton, MyImageWidget


class ButtonSolve(MDFillRoundFlatIconButton):
    dict_param = {
        "solve": {
            "text": "Solve",
            "icon": "crop-free",
        },
        "unfreeze": {
            "text": "Unfreeze",
            "icon": "restart",
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**self.dict_param["solve"], **kwargs)
        self.__is_freeze = False

    @property
    def is_freeze(self):
        return self.__is_freeze

    def override_on_release(self):
        super(ButtonSolve, self).on_release()
        self.__is_freeze = not self.__is_freeze
        param_name = "unfreeze" if self.__is_freeze else "solve"
        params = self.dict_param[param_name]
        self.text = params["text"]
        self.icon = params["icon"]


class ScreenLiveSolver(MDBottomNavigationItem):
    def __init__(self, solver, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Live Solver'
        self.icon = 'camera'

        # -- UI
        self.__big_layout = MDBoxLayout(padding=20, spacing=20, orientation="vertical")
        self.add_widget(self.__big_layout)

        # - Video Stream
        self.__image_layout = MyImageWidget(size_hint_y="0.75", id="image")
        self.__big_layout.add_widget(self.__image_layout)
        self.__video_stream = MyKivyCamera(auto_start=True)
        self.__fps = 30
        self.__last_image_read = None

        Clock.schedule_interval(self.read_video_stream, 1.0 / self.__fps)

        # - Option Layout
        self.__layout_options = RelativeLayout(size_hint_y="0.25")

        self.__check_group = "Hint"
        self.__button_hint_mode = MyToggleButton(text="Hint Mode", pos_hint={"center_x": 0.1, "center_y": 0.5},
                                                 group=self.__check_group)
        self.__button_full_solve = MyToggleButton(text="Full Solve", pos_hint={"center_x": 0.25, "center_y": 0.5},
                                                  group=self.__check_group)

        self.__button_full_solve.set_state(True)

        self.__layout_options.add_widget(self.__button_hint_mode)
        self.__layout_options.add_widget(self.__button_full_solve)

        self.__solve_unfreeze_button = ButtonSolve(pos_hint={"center_x": 0.8, "center_y": 0.5})
        self.__solve_unfreeze_button.bind(on_release=self.callback_solve_unfreeze_button)

        self.__layout_options.add_widget(self.__solve_unfreeze_button)

        self.__big_layout.add_widget(self.__layout_options)

        # - Solver
        self.__solver = solver

    def read_video_stream(self, _dt):
        if self.parent is None:
            self.__video_stream.pause()
            return
        else:
            self.__video_stream.resume()

        frame = self.__video_stream.last_image_read

        if frame is None:
            return

        if not self.__solve_unfreeze_button.is_freeze:
            self.__last_image_read = frame
            self.set_new_image(frame)

    def set_new_image(self, frame):
        self.__image_layout.image = frame

    def should_give_only_hint(self):
        return self.__button_hint_mode.is_active()

    def callback_solve_unfreeze_button(self, *_args):
        self.__solve_unfreeze_button.override_on_release()
        is_freeze = self.__solve_unfreeze_button.is_freeze

        if is_freeze:
            self.solve()

    def solve(self):
        hint_mode = self.should_give_only_hint()
        solved_frame = self.__solver.solve_1_img(img=self.__last_image_read, hint_mode=hint_mode)
        self.display_solved(solved_frame)

    def display_solved(self, solved_frame):
        self.__last_image_read = solved_frame
        self.set_new_image(solved_frame)


if __name__ == '__main__':
    from kivymd.app import MDApp
    from src.SolverVR import SolverVR


    class LiveSolverApp(MDApp):
        def build(self):
            return ScreenLiveSolver(SolverVR())


    LiveSolverApp().run()
