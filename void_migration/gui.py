from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivymd.app import MDApp as App
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.selectioncontrol import MDCheckbox as CheckBox
from kivymd.uix.label import MDLabel as Label
from kivymd.uix.slider import MDSlider as Slider
from kivymd.uix.textfield import MDTextField as TextInput
from kivymd.uix.menu import MDDropdownMenu as DropdownMenu
from kivymd.uix.dropdownitem import MDDropDownItem as DropDownItem
from kivymd.uix.button.button import MDRaisedButton as Button
from kivy.cache import Cache
from kivy.config import Config
from kivy.logger import Logger, LOG_LEVELS
import void_migration.params as params
from void_migration.main import time_march


import sys
import os
import signal
import multiprocessing
from functools import partial

os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"

# Set log level to warning
Config.set("kivy", "log_level", "warning")
Logger.setLevel(LOG_LEVELS["warning"])


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
        self.menus = {}

    def build(self):
        self.title = "Void Migration"
        main_layout = BoxLayout(orientation="horizontal")
        param_layout = BoxLayout(orientation="vertical", size_hint_x=0.3, padding=20)

        for key, limits in self.data["gui"].items():
            value = getattr(self.p, key)
            if key not in ["save", "videos", "plot"]:
                param_layout.add_widget(Label(text=limits["title"]))
                if limits["dtype"] == "bool":
                    input_widget = CheckBox(active=value)
                    input_widget.bind(active=partial(self.update_param, key=key))
                elif limits["dtype"] == "int":
                    input_widget = Slider(
                        min=limits["min"], max=limits["max"], step=limits["step"], value=value
                    )
                    input_widget.bind(value=partial(self.update_param, key=key))
                elif limits["dtype"] == "float":
                    input_widget = Slider(
                        min=limits["min"], max=limits["max"], step=limits["step"], value=value
                    )
                    input_widget.bind(value=partial(self.update_param, key=key))
                elif limits["dtype"] == "str":
                    input_widget = TextInput(text=value)
                    input_widget.bind(text=partial(self.update_param, key=key))
                elif limits["dtype"] == "select":
                    input_widget = DropDownItem()
                    menu_items = [
                        {
                            "text": option,
                            "viewclass": "OneLineListItem",
                            "on_release": lambda x=option, k=key: self.menu_callback(x, k),
                        }
                        for option in limits["options"]
                    ]
                    dropdown_menu = DropdownMenu(
                        caller=input_widget,
                        items=menu_items,
                        width_mult=4,
                    )
                    self.menus[key] = dropdown_menu
                    input_widget.bind(on_release=lambda x, k=key: self.menus[k].open())
                    input_widget.set_item(value)
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

    def menu_callback(self, text_item, key):
        dropdown_item = getattr(self, f"input_{key}", None)
        if isinstance(dropdown_item, DropDownItem):
            dropdown_item.set_item(text_item)
            self.update_param(dropdown_item, key=key)
            self.menus[key].dismiss()

    def update_param(self, instance, *args, **kwargs):
        key = kwargs.get("key")
        if isinstance(instance, CheckBox):
            value = instance.active
        elif isinstance(instance, DropDownItem):
            value = instance.current_item
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

        setattr(self.p, key, value)
        print(f"Updated {key} to {value}")

    def update_image(self):
        # Check for updates from the queue
        while not self.queue.empty():
            t = self.queue.get()
            LATEST_IMAGE = p.folderName + f"{self.p.view}_{str(t).zfill(6)}.png"
            if os.path.exists(LATEST_IMAGE):
                try:
                    Cache.remove("kv.image")
                    Cache.remove("kv.texture")
                    core_img = CoreImage(LATEST_IMAGE, ext="png")
                    core_img.texture.min_filter = "nearest"
                    core_img.texture.mag_filter = "nearest"
                    self.img.texture = core_img.texture
                    self.img.canvas.ask_update()  # Force the image widget to redraw
                except Exception as e:
                    print(f"Error updating image: {e}")

    def start_time_march(self, instance):
        if self.process is not None:
            self.stop_time_march(instance)
        self.process = multiprocessing.Process(
            target=run_time_march, args=(self.p, self.queue, self.stop_event)
        )
        self.process.start()

    def stop_time_march(self, instance):
        if self.process is not None:
            self.stop_event.set()
            self.process.join()
            self.stop_event.clear()
            print("Process terminated")

    def on_stop(self):
        self.stop_time_march(None)
        # Kill the process when the app closes
        os.kill(self.process.pid, signal.SIGTERM)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    if len(sys.argv) < 2:
        filename = params.resource_path("json/gui.json5")
    else:
        filename = sys.argv[1]
    with open(filename, "r") as f:
        data, p = params.load_file(f)
    p.concurrent_index = 0

    VoidMigrationApp(data, p).run()
