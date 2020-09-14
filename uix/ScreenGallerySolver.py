import cv2

# from kivy.core.window import Window

from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDRoundFlatIconButton
from uix.pimped_widgets import MyImageWidget


class ScreenGallerySolver(MDBottomNavigationItem):
    def __init__(self, solver, **kwargs):
        super().__init__(**kwargs)
        self.text = 'Gallery Solver'
        self.icon = 'folder-image'

        # -- UI
        self.__big_layout = MDBoxLayout(padding=20, spacing=20, orientation="vertical")
        self.add_widget(self.__big_layout)
        self.__big_layout.add_widget(MDLabel(text='Gallery Solver', halign='center', font_style="H5",
                                             size_hint_y=0.15))

        # - Image Layout
        self.__image_layout = MyImageWidget(size_hint_y="0.75", id="image",
                                            source='uix/assets/AI.png')

        self.__big_layout.add_widget(self.__image_layout)

        # - File Manager for image selection
        # Window.bind(on_keyboard=self.events)

        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            # preview=True,
        )
        self.__image_selection_button = MDRoundFlatIconButton(on_release=self.file_manager_open,
                                                              pos_hint={"top": 0.4, "center_x": 0.5})
        self.__image_selection_button.text = "Select An Image !"
        self.__image_selection_button.icon = "image"

        # self.__actual_image_path = None

        self.__big_layout.add_widget(self.__image_selection_button)

        # - Solver
        self.__solver = solver

    def file_manager_open(self, *_):
        self.file_manager.show(".")  # (os.path.expanduser('~'))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        self.exit_manager()
        if not path.endswith((".jpg", ".png")):
            return

        frame = cv2.imread(path)
        if frame is None:
            return
            # self.__actual_image_path = path

        self.solve_n_display(frame)

    def exit_manager(self, *_args):
        self.manager_open = False
        self.file_manager.close()

    # def events(self, _instance, keyboard, _keycode, _text, _modifiers):
    #     """
    #     Called when buttons are pressed on the mobile device.
    #     """
    #     print(keyboard)
    #     if keyboard in (1001, 27):
    #         if self.manager_open:
    #             self.file_manager.back()
    #     return True

    def solve_n_display(self, frame):
        hint_mode = False
        solved_frame = self.__solver.solve_1_img(img=frame, hint_mode=hint_mode)
        self.set_new_image(solved_frame)

    def set_new_image(self, frame):
        self.__image_layout.image = frame


if __name__ == '__main__':
    from kivymd.app import MDApp
    from src.SolverVR import SolverVR


    class GallerySolverApp(MDApp):
        def build(self):
            return ScreenGallerySolver(SolverVR())


    GallerySolverApp().run()
