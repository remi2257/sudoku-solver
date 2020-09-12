import numpy as np

from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.relativelayout import MDRelativeLayout
from kivymd.uix.selectioncontrol import MDCheckbox, MDSwitch
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.uix.image import Image

from kivymd.app import MDApp
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivymd.uix.button import MDFillRoundFlatButton
from kivymd.uix.slider import MDSlider
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.menu import MDDropdownMenu
from uix.kivy_useful_func import *


class MyToggleButton(MDFillRoundFlatButton, MDToggleButton):
    def __init__(self, allow_no_selection=False, **kwargs):
        super().__init__(allow_no_selection=allow_no_selection, **kwargs)
        self.background_down = MDApp.get_running_app().theme_cls.primary_dark
        self.background_normal = MDApp.get_running_app().theme_cls.primary_light
        self.state = "down"
        self.state = "normal"

    def is_active(self):
        return self.state == "down"

    def set_state(self, state):
        if not type(state) == bool:
            return
        self.state = "down" if state else "normal"


class Check(MDCheckbox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = None, None
        self.size = dp(48), dp(48)


class CheckWithText(MDRelativeLayout):
    def __init__(self, text, **kwargs):
        super().__init__()

        self.check_box = Check(pos_hint={"x": 0.}, size_hint_x=0.3, **kwargs)
        self.add_widget(self.check_box)
        self.add_widget(MDLabel(text=text, size_hint_x=0.7,
                                pos_hint={"right": 1.}))

    @property
    def pushed(self):
        return self.check_box.state == "down"


class SwitchWithText(MDBoxLayout):
    def __init__(self, text, **kwargs):
        super().__init__(orientation='horizontal', spacing=20, **kwargs)
        # super().__init__(size=(200, 100), size_hint=(None, None))
        self.switch = MDSwitch(pos_hint={"center_x": 0.2, 'center_y': 0.5})
        self.switch.width = dp(48)
        self.add_widget(self.switch)
        self.add_widget(MDLabel(pos_hint={'center_y': 0.5},
                                text=text, valign="center"))

    def is_active(self):
        return self.switch.active


class SliderWithText(MDRelativeLayout):
    def __init__(self, text_format, size_hint_y=1.0, **kwargs):
        super().__init__(size_hint_y=size_hint_y)
        self.slider = MDSlider(size_hint=(0.7, 1), **kwargs)
        self.add_widget(self.slider)
        self.text_format = text_format
        self.label = MDLabel(size_hint=(0.3, 1), pos_hint={"right": 1.})
        self.set_text_value()
        self.add_widget(self.label)

        Clock.schedule_interval(self.set_text_value, 0.1)

    def is_active(self):
        return self.slider.active

    @property
    def value(self):
        return int(self.slider.value)

    def set_text_value(self, *_):
        self.label.text = self.text_format.format(self.value)


class MyDropdown(MDDropDownItem):
    def __init__(self, list_items, icon, first_item_chosen=None, **kwargs):
        super(MyDropdown, self).__init__(**kwargs)
        menu_items = [{"icon": icon, "text": str(item)} for item in list_items]
        self.menu = MDDropdownMenu(caller=self,
                                   items=menu_items, position="auto",
                                   width_mult=5)

        self.menu.bind(on_release=self.set_item_dropdown)
        self.bind(on_release=lambda _: self.menu.open())

        if first_item_chosen is None:
            first_item_chosen = str(list_items[0])
        self.set_item(first_item_chosen)

    def set_item_dropdown(self, _instance_menu, instance_menu_item):
        selected_item = instance_menu_item.text
        self.menu.dismiss()
        self.set_item(selected_item)


class ModelDropdown(MyDropdown):
    def __init__(self, models_dict, chosen_model_name, **kwargs):
        super(ModelDropdown, self).__init__(list_items=list(models_dict.keys()),
                                            icon="sitemap", first_item_chosen=chosen_model_name,
                                            **kwargs)


class MyImageWidget(Image):
    def __init__(self, **kwargs):
        super(MyImageWidget, self).__init__(**kwargs)

    @property
    def image(self):
        return convert_texture_to_opencv(self.texture)

    @image.setter
    def image(self, frame):
        if frame is None:
            frame = 255 * np.ones((256, 256))
        self.texture = convert_opencv_to_texture(frame)


class IntegerTextField:
    pass
