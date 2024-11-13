import tkinter as tk
from tkinter import ttk
import sys
if sys.platform == 'win32':
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

def should_hide_device(name):
    name = name.lower()
    return \
        'cable in' in name or \
        'blackhole' in name or \
        'primary sound driver' in name or \
        'microsoft sound mapper' in name

def get_devices():
    pa = pyaudio.PyAudio()
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    outputs = [dev['name'] for dev in devices if dev['maxOutputChannels'] > 0 and not should_hide_device(dev['name'])]
    pa.terminate()
    return list(set(outputs))

class ZenGui:
    def __init__(self, initial_zen_mode_state):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        # root.overrideredirect(True)
        self.root.title("Zen Mode Control")
        self.root.geometry("320x320")  # Set a fixed size for the window
        self.root.iconphoto(True, tk.PhotoImage(file='./img/zen-256.png'))  # Set icon

        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 10), padding=2)
        style.configure("TCheckbutton", font=("Helvetica", 12), padding=5)

        self.zen_mode_checkbox = ttk.Checkbutton(self.root, text="Zen Mode")
        self.zen_mode_checkbox.pack(pady=1)
        self.zen_mode_checkbox.invoke()
        if not initial_zen_mode_state:
            self.zen_mode_checkbox.invoke()
        self.zen_mode_checkbox.bind("<Button-1>", self.evt_checkbox)

        # Listbox for device selection
        self.device_listbox_frame = tk.Frame(self.root)
        self.device_listbox_frame.pack(pady=1, padx=1)
        self.device_listbox = tk.Listbox(self.device_listbox_frame, width=40, height=8, font=("Helvetica", 10), selectmode=tk.SINGLE)
        self.device_listbox.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        scrollbar = tk.Scrollbar(self.device_listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.device_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.device_listbox.yview)
        for device in get_devices():
            self.device_listbox.insert(tk.END, device)
        
        # self.device_listbox.selection_set(0)
        self.device_listbox.bind("<<ListboxSelect>>", self.evt_device_list_select)

        # Volume control for Device 1
        self.music_volume_label = ttk.Label(self.root, text="Music Volume")
        self.music_volume_label.pack(pady=1)

        self.music_volume_slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, value=100, length=250)
        self.music_volume_slider.pack(pady=1, padx=20)
        self.music_volume_slider.bind("<Button-1>", self.evt_update_slider)
        self.music_volume_slider.bind("<B1-Motion>", self.evt_update_slider)

        # Volume control for Device 2
        self.vocals_volume_label = ttk.Label(self.root, text="Vocals Volume")
        self.vocals_volume_label.pack(pady=1)

        self.vocals_volume_slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, value=0, length=250)
        self.vocals_volume_slider.pack(pady=1, padx=20)
        self.vocals_volume_slider.bind("<Button-1>", self.evt_update_slider)
        self.vocals_volume_slider.bind("<B1-Motion>", self.evt_update_slider)

        # Copyright
        self.copyright_label = ttk.Label(self.root, text="Copyright © 2024 Zen Mode - @high_byte")
        self.copyright_label.pack(pady=1)

    def evt_checkbox(self, event):
        state = self.zen_mode_checkbox.instate(['selected'])
        # this is the prev state
        self.on_zen_mode_change(not state)

    def evt_update_slider(self, event):
        # Calculate the new value based on mouse position
        value = (event.x / event.widget.winfo_width()) * event.widget['to']
        event.widget.set(value)
        volume = max(0.0, min(1.0, value / 100.0))
        self.on_volume_change('music' if event.widget == self.music_volume_slider else 'vocals', volume)

    def evt_device_list_select(self, event):
        index = self.device_listbox.curselection()[0]

        # Update the listbox with the new device names
        self.device_listbox.delete(0, tk.END)  # Clear the listbox
        devices = get_devices()
        for device in devices:
            self.device_listbox.insert(tk.END, device)

        # Select item by index
        self.device_listbox.selection_set(index)
        if index < len(devices):
            self.on_device_select(devices[index])

    def on_zen_mode_change(self, value):
        print('on_zen_mode_change', value)

    def on_volume_change(self, type, value):
        print('on_volume_change', type, value)

    def on_device_select(self, name):
        print('on_device_select', name)

if __name__ == "__main__":
    app = ZenGui()
    app.root.mainloop()
