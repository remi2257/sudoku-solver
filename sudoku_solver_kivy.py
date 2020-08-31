import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button

from ui.LiveSolverScreen import LiveSolverScreen
from ui.GallerySolverScreen import GallerySolverScreen

kivy.require('1.9.0')


class ManualSolverScreen(Screen):
    pass


class HelpScreen(Screen):
    pass


class AboutScreen(Screen):
    pass


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)

        float_layout = FloatLayout()

        float_layout.add_widget(
            Button(id="button_help", text="Help",
                   pos_hint={"top": 0.95}, size_hint_y=0.1,
                   on_release=self.button_help)
        )

        float_layout.add_widget(
            Button(id="button_live_solver", text="Live Solver",
                   pos_hint={"top": 0.8}, size_hint_y=0.1,
                   on_release=self.button_live_solver)
        )

        float_layout.add_widget(
            Button(id="button_gallery_solver", text="Gallery Solver",
                   pos_hint={"top": 0.65}, size_hint_y=0.1,
                   on_release=self.button_gallery_solver)
        )

        float_layout.add_widget(
            Button(id="button_manual_solver", text="Manual Solver",
                   pos_hint={"top": 0.5}, size_hint_y=0.1,
                   on_release=self.button_manual_solver)
        )

        float_layout.add_widget(
            Button(id="button_about", text="About",
                   pos_hint={"top": 0.3}, size_hint_y=0.1,
                   on_release=self.button_about)
        )

        self.add_widget(float_layout)

    def button_help(self, _button_state):
        self.manager.transition.direction = "down"
        self.manager.current = "help"

    def button_live_solver(self, _button_state):
        self.manager.transition.direction = "right"
        self.manager.current = "live_solver"

        # index_live_solver = self.manager.screen_names.index("live_solver")
        self.manager.current_screen.start_stream()

    def button_gallery_solver(self, _button_state):
        self.manager.transition.direction = "left"
        self.manager.current = "gallery_solver"

    def button_manual_solver(self, _button_state):
        self.manager.transition.direction = "left"
        self.manager.current = "manual_solver"

    def button_about(self, _button_state):
        self.manager.transition.direction = "up"
        self.manager.current = "about"


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)

        self.add_widget(MainScreen(name="main", id="main"))
        self.add_widget(HelpScreen(name="help", id="help"))
        self.add_widget(LiveSolverScreen(name="live_solver", id="live_solver"))
        self.add_widget(GallerySolverScreen(name="gallery_solver", id="gallery_solver"))
        self.add_widget(ManualSolverScreen(name="manual_solver", id="manual_solver"))
        self.add_widget(AboutScreen(name="about", id="about"))


class SudokuSolverApp(App):
    def build(self):
        return WindowManager()


if __name__ == "__main__":
    SudokuSolverApp().run()
