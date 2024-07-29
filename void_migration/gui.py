import sys
import os
import multiprocessing
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivymd.app import MDApp as App
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.selectioncontrol import MDCheckbox as CheckBox
from kivymd.uix.label import MDLabel as Label
from kivymd.uix.slider import MDSlider as Slider
from kivymd.uix.textfield import MDTextField as TextInput
from kivymd.uix.button.button import MDRaisedButton as Button

import params
from void_migration import time_march


def run_time_march(p, queue, stop_event, *args):
    p.set_defaults()
    time_march(p, queue, stop_event)


class VoidMigrationApp(App):
    def __init__(self, data, p, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.p = p
        self.halt = False
        self.queue = multiprocessing.Queue()
        self.process = None
        self.stop_event = multiprocessing.Event()

    def build(self):
        main_layout = BoxLayout(orientation="horizontal")
        param_layout = BoxLayout(orientation="vertical", size_hint_x=0.3)

        # for key, value in self.data.items():
        #     if key not in ["save", "videos", "plot"]:
        #         param_layout.add_widget(Label(text=key))
        #         if isinstance(value, bool):
        #             input_widget = TextInput(text=str(value))
        #         elif isinstance(value, (int, float)):
        #             input_widget = Slider(min=0, max=100, value=value)
        #         elif isinstance(value, str):
        #             input_widget = TextInput(text=value)
        #         else:
        #             raise ValueError(f"Unsupported type: {type(value)} for key: {key}")
        #         param_layout.add_widget(input_widget)
        #         setattr(self, f"input_{key}", input_widget)

        for key, value in self.data.items():
            if key not in ["save", "videos", "plot"]:
                param_layout.add_widget(Label(text=key))
                if isinstance(value, bool):
                    input_widget = CheckBox(active=value)
                    input_widget.bind(active=lambda instance, key=key: self.update_param(instance, key))
                elif isinstance(value, (int, float)):
                    input_widget = Slider(min=0, max=100, value=value)
                    input_widget.bind(on_value=lambda instance, key=key: self.update_param(instance, key))
                elif isinstance(value, str):
                    input_widget = TextInput(text=value)
                    input_widget.bind(on_text=lambda instance, key=key: self.update_param(instance, key))
                else:
                    raise ValueError(f"Unsupported type: {type(value)} for key: {key}")
                param_layout.add_widget(input_widget)
                setattr(self, f"input_{key}", input_widget)

        buttons = BoxLayout(
            orientation="horizontal",
            size_hint_x=1.0,
            size_hint_y=None,
            height=80,
            spacing=20,
            # padding=[50, 0],
        )

        run_button = Button(text="Start", size_hint_x=0.5, size_hint_y=None, height=80)
        run_button.bind(on_press=self.start_time_march)
        buttons.add_widget(run_button)

        stop_button = Button(text="Stop", size_hint_x=0.5)
        stop_button.bind(on_press=self.stop_time_march)
        buttons.add_widget(stop_button)

        img_layout = BoxLayout(orientation="vertical")
        img_layout.add_widget(buttons)

        self.img = Image(allow_stretch=True, keep_ratio=True)
        img_layout.add_widget(self.img)

        # Add parameter and image layouts to the main layout
        main_layout.add_widget(param_layout)
        main_layout.add_widget(img_layout)

        Clock.schedule_interval(lambda dt: self.update_image(), 0.1)  # Start watching image directory

        return main_layout

    def update_param(self, instance, key):
        if isinstance(instance, CheckBox):
            value = instance.active
        elif isinstance(instance, TextInput):
            value = instance.text
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        elif isinstance(instance, Slider):
            value = instance.value
        print(f"Updating {key} to {value}")
        setattr(self, key, value)
        print(f"Updated {key} to {value}")

    def update_image(self):
        # Check for updates from the queue
        while not self.queue.empty():
            LATEST_IMAGE = self.queue.get()
            if os.path.exists(LATEST_IMAGE):
                try:
                    core_img = CoreImage(LATEST_IMAGE, ext="png")
                    core_img.texture.min_filter = "nearest"
                    core_img.texture.mag_filter = "nearest"
                    self.img.texture = core_img.texture
                    self.img.canvas.ask_update()  # Force the image widget to redraw
                    # print(f"Updated image: {LATEST_IMAGE}")
                except Exception as e:
                    print(f"Error updating image: {e}")

    def start_time_march(self, instance):
        self.process = multiprocessing.Process(
            target=run_time_march, args=(self.p, self.queue, self.stop_event)
        )
        self.process.start()

    def stop_time_march(self, instance):
        if self.process is not None:
            self.stop_event.set()
            # self.process.terminate()
            self.process.join()
            self.stop_event.clear()
            print("Process terminated")


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        data, p = params.load_file(f)
    p.concurrent_index = 0
    p.gui = True

    VoidMigrationApp(data, p).run()
