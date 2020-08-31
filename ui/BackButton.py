from kivy.uix.button import Button


class BackButton(Button):
    def __init__(self, transition_dir="right", **kwargs):
        super(BackButton, self).__init__(**kwargs)
        self.transition_dir = transition_dir

    def on_release(self):
        self.manager.transition.direction = self.transition_dir
        self.manager.current = "main"
