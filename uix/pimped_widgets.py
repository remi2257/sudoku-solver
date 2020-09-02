from kivymd.uix.label import MDLabel
from kivy.uix.relativelayout import RelativeLayout
from kivymd.uix.selectioncontrol import MDCheckbox, MDSwitch

from kivy.metrics import dp


class Check(MDCheckbox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = None, None
        self.size = dp(48), dp(48)


class CheckWithText(RelativeLayout):
    def __init__(self, text, size_hint_x, id_, **kwargs):
        super().__init__(size_hint_x=size_hint_x, id=id_)

        self.check_box = Check(pos_hint={'x': 0., 'center_y': 0.5}, **kwargs)
        self.add_widget(self.check_box)
        self.add_widget(MDLabel(text=text, pos_hint={'x': 0.2, 'center_y': 0.5}))

    def is_active(self):
        return self.check_box.active


class SwitchWithText(RelativeLayout):
    def __init__(self, text, size_hint_x, id_, **kwargs):
        super().__init__(size_hint_x=size_hint_x, id=id_)
        # super().__init__(size=(200, 100), size_hint=(None, None))
        self.switch = MDSwitch(pos_hint={'center_x': 0.2, 'center_y': 0.5}, **kwargs)
        self.add_widget(self.switch)
        self.add_widget(MDLabel(text=text, pos_hint={'x': 0.4, 'center_y': 0.5}))

    def is_active(self):
        return self.switch.active

class ButtonWithText(RelativeLayout):
    def __init__(self, text, size_hint_x, id_, **kwargs):
        super().__init__(size_hint_x=size_hint_x, id=id_)
        # super().__init__(size=(200, 100), size_hint=(None, None))
        self.switch = MDSwitch(pos_hint={'center_x': 0.2, 'center_y': 0.5}, **kwargs)
        self.add_widget(self.switch)
        self.add_widget(MDLabel(text=text, pos_hint={'x': 0.4, 'center_y': 0.5}))

    def is_active(self):
        return self.switch.active
