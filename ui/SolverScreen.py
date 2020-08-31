from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen

from tensorflow.keras.models import load_model
from src.main_single_img import process_single_img

model_default_name = 'model/my_model.h5'


class SolverScreen(Screen):

    def __init__(self, **kwargs):
        super(SolverScreen, self).__init__(**kwargs)

        self.btn_solve_unfreeze = Button(id="btn_solve_unfreeze", text="Solve",
                                         pos_hint={"top": 0.5}, size_hint_y=0.1)

        self.add_widget(self.btn_solve_unfreeze)

        self.btn_full_solve = ToggleButton(id="btn_full_solve", text="Full Solve", group="solve",
                                           state="down", pos_hint={"x": 0.2, "y": 0.1},
                                           size_hint=(0.1, 0.1))
        self.add_widget(self.btn_full_solve)

        self.btn_hint = ToggleButton(id="btn_hint", text="Hint", group="solve",
                                     pos_hint={"x": 0.3, "y": 0.1},
                                     size_hint=(0.1, 0.1))
        self.add_widget(self.btn_hint)

        self.model = load_model(model_default_name)

    @property
    def target_image(self):
        raise NotImplementedError

    def solve(self):
        frame = self.target_image
        hint_mode = self.should_give_only_hint()
        solved_frame = process_single_img(frame, self.model, hint_mode=hint_mode)
        self.display_solved(solved_frame)

    def display_solved(self, opencv_frame):
        raise NotImplementedError

    def should_give_only_hint(self):
        return self.btn_hint.state == "down"
