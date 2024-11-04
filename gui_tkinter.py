import pyaudiowpatch as pyaudio
import tkinter as tk
from tkinter import ttk

pa = pyaudio.PyAudio()

def get_devices():
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    outputs = [dev['name'] for dev in devices if dev['maxOutputChannels'] > 0 and 'CABLE In' not in dev['name']]
    return list(set(outputs))

class VolumeControlApp:
    def __init__(self, root):
        self.devices = get_devices()
        self.root = root
        self.root.title("Zen Mode Control")
        self.root.geometry("340x420")  # Set a fixed size for the window
        self.root.iconphoto(True, tk.PhotoImage(file='./img/zen-256.png'))  # Set icon

        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 10), padding=2)
        style.configure("TCheckbutton", font=("Helvetica", 12), padding=5)

        self.zen_mode_checkbox = ttk.Checkbutton(root, text="Zen Mode", variable=tk.IntVar(value=1))
        self.zen_mode_checkbox.pack(pady=1)
        self.zen_mode_checkbox.invoke()
        self.zen_mode_checkbox.bind("<Button-1>", lambda e: print('zen mode'))

        # Volume control for Device 1
        self.music_volume_label = ttk.Label(root, text="Music Volume")
        self.music_volume_label.pack(pady=1)

        self.music_volume_slider = ttk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, value=100, length=250)
        self.music_volume_slider.pack(pady=1, padx=20)
        self.music_volume_slider.bind("<Button-1>", self.update_slider)
        self.music_volume_slider.bind("<B1-Motion>", self.update_slider)

        # Volume control for Device 2
        self.vocals_volume_label = ttk.Label(root, text="Vocals Volume")
        self.vocals_volume_label.pack(pady=1)

        self.vocals_volume_slider = ttk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, value=0, length=250)
        self.vocals_volume_slider.pack(pady=1, padx=20)
        self.vocals_volume_slider.bind("<Button-1>", self.update_slider)
        self.vocals_volume_slider.bind("<B1-Motion>", self.update_slider)

        # Listbox for device selection
        self.device_listbox_frame = tk.Frame(root)
        self.device_listbox_frame.pack(pady=1, padx=1)
        self.device_listbox = tk.Listbox(self.device_listbox_frame, width=40, height=15, font=("Helvetica", 10))
        self.device_listbox.pack(side=tk.LEFT)
        scrollbar = tk.Scrollbar(self.device_listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.device_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.device_listbox.yview)
        self.device_listbox.bind("<<ListboxSelect>>", self.on_device_select)

        # Simulate fetching device names (replace with actual device fetching logic)
        self.device_listbox.delete(0, tk.END)  # Clear the listbox
        for device in self.devices:
            self.device_listbox.insert(tk.END, device)


    def update_slider(self, event):
        # Calculate the new value based on mouse position
        value = (event.x / event.widget.winfo_width()) * event.widget['to']
        event.widget.set(value)

    def on_device_select(self, event):
        index = self.device_listbox.curselection()
        selected_device = self.device_listbox.get(index)
        print(f"Selected device: {selected_device}")

        self.devices = get_devices()

        # Update the listbox with the new device names
        self.device_listbox.delete(0, tk.END)  # Clear the listbox
        for device in self.devices:
            self.device_listbox.insert(tk.END, device)

        # Select item by index
        self.device_listbox.selection_set(index)

if __name__ == "__main__":
    root = tk.Tk()
    app = VolumeControlApp(root)
    root.mainloop()