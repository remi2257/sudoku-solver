from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.label import MDLabel


class ScreenAbout(MDBottomNavigationItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = 'About'
        self.icon = 'help'

        self.add_widget(MDLabel(text='Work In Progress', halign='center'))
