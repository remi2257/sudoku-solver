from .SolverScreen import SolverScreen
from uix.MyKivyCamera import MyKivyCamera, convert_opencv_to_texture
# from.BackButton import BackButton
from kivy.uix.button import Button


class LiveSolverScreen(SolverScreen):

    def __init__(self, **kwargs):
        super(LiveSolverScreen, self).__init__(**kwargs)
        self.camera = MyKivyCamera(auto_start=False,
                                   pos_hint={"top": 1}, size_hint_y=0.8)
        self.add_widget(self.camera)

        self.button_solve_unfreeze = Button(id="button_solve_unfreeze", text="Solve",
                                            pos_hint={"x": 0.6, "y": 0.1}, size_hint=(0.2, 0.1),
                                            on_release=self.cb_button_solve_unfreeze)
        self.add_widget(self.button_solve_unfreeze)

        self.add_widget(
            Button(id="button_back", text="Back",
                   pos_hint={"y": 0}, size_hint_y=0.1,
                   on_release=self.button_back)
        )

    @property
    def target_image(self):
        return self.camera.last_image_read

    def display_solved(self, opencv_frame):
        self.camera.freeze()
        self.camera.texture = convert_opencv_to_texture(opencv_frame)

    def start_stream(self):
        self.camera.start_capture()

    def stop_stream(self):
        self.camera.stop_capture()

    def cb_button_solve_unfreeze(self, _button_state):
        if _button_state.text == "Solve":
            _button_state.text = "Unfreeze"
            self.solve()
        else:
            _button_state.text = "Solve"
            self.camera.unfreeze()

    def button_back(self, _button_state):
        self.stop_stream()
        self.manager.transition.direction = "left"
        self.manager.current = "main"
