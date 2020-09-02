from kivymd.uix.bottomnavigation import MDBottomNavigationItem
from kivymd.uix.label import MDLabel


class ScreenReporting(MDBottomNavigationItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.icon = 'tablet-dashboard'

        self.add_widget(MDLabel(text='Reporting', halign='center'))
