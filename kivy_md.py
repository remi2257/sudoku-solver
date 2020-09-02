from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation
from uix.ScreenAbout import ScreenAbout
from uix.LiveSolverScreen import LiveSolverScreen
from uix.ScreenReporting import ScreenReporting
from uix.ScreenTrain import ScreenTrain
from kivymd.color_definitions import text_colors


class BottomNav(MDBottomNavigation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.text_color_active = text_colors["Red"]["50"]
        # self.text_color_normal = [1, 0, 1, 1]
        # self.panel_color = [.2, .2, .2, 1]
        self.add_widget(LiveSolverScreen(name="Predict", id="Predict"))
        self.add_widget(ScreenAbout(name="About", id="About"))


class SudokuAppApp(MDApp):
    def build(self):
        return BottomNav()


SudokuAppApp().run()
