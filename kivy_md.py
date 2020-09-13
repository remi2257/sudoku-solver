from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation
from uix.ScreenAbout import ScreenAbout
from uix.ScreenLiveSolver import ScreenLiveSolver
from uix.ScreenGallerySolver import ScreenGallerySolver

from src.SolverVR import SolverVR


class BottomNav(MDBottomNavigation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        solver = SolverVR()

        self.add_widget(ScreenLiveSolver(solver=solver, name="Live Solver", id="Live Solver"))
        self.add_widget(ScreenGallerySolver(solver=solver, name="Gallery Solver", id="Gallery Solver"))
        self.add_widget(ScreenAbout(name="About", id="About"))


class SudokuAppApp(MDApp):
    def build(self):
        return BottomNav()


if __name__ == '__main__':
    SudokuAppApp().run()
