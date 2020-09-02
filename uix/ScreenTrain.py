from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.label import MDLabel


class ScreenTrain(MDBottomNavigationItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.icon = 'database-sync'

        self.add_widget(MDLabel(text='Train', halign='center'))
