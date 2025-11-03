from tkinter import Tk, Frame, Label, Entry, Button, Text, Scrollbar, END
import threading
import sys
import io
import pyvisa
from oscilloscope import find_instrument, get_waveform_data, save_waveform_csv

class OscilloscopeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Oscilloscope GUI")
        master.geometry("400x300")
        master.attributes("-topmost", True)

        self.resource_string = Entry(master, width=30)
        self.resource_string.insert(0, 'ASRL5::INSTR')
        self.resource_string.pack(pady=10)

        self.start_stop_button = Button(master, text="Start/Stop Communication", command=self.toggle_communication)
        self.start_stop_button.pack(pady=5)

        self.get_data_button = Button(master, text="Get Waveform Data", command=self.get_waveform_data)
        self.get_data_button.pack(pady=5)

        self.terminal_frame = Frame(master)
        self.terminal_frame.pack(pady=10)

        self.terminal = Text(self.terminal_frame, height=10, width=50)
        self.terminal.pack(side="left")

        self.scrollbar = Scrollbar(self.terminal_frame, command=self.terminal.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.terminal.config(yscrollcommand=self.scrollbar.set)

        self.original_stdout = sys.stdout
        self.rm = pyvisa.ResourceManager()
        self.redirect_output()

        self.scope = None
        self.is_connected = False

    def redirect_output(self):
        class StdoutRedirector:
            def __init__(self, text_widget, original):
                self.text_widget = text_widget
                self.original = original
            def write(self, s):
                if s:
                    self.text_widget.insert(END, s)
                    self.text_widget.see(END)
                self.original.write(s)
            def flush(self):
                try:
                    self.original.flush()
                except Exception:
                    pass
        sys.stdout = StdoutRedirector(self.terminal, self.original_stdout)

    def toggle_communication(self):
        if self.is_connected:
            if self.scope:
                try:
                    self.scope.close()
                except Exception:
                    pass
                print("Connection closed.")
            self.is_connected = False
        else:
            resource_string = self.resource_string.get().strip()
            threading.Thread(target=self._connect_thread, args=(resource_string,), daemon=True).start()
            print("Connecting...")
    def _connect_thread(self, resource_string):
        inst = find_instrument(self.rm, resource_string)
        if inst:
            self.scope = inst
            self.is_connected = True
            self.master.after(0, lambda: print("Connected."))
        else:
            self.master.after(0, lambda: print("Connection failed."))

    def get_waveform_data(self):
        if self.is_connected and self.scope:
            voltages, time_interval = get_waveform_data(self.scope, 1)
            if voltages is not None:
                file = save_waveform_csv(voltages, time_interval)
                if file:
                    print(f"Waveform saved as {file}")
                else:
                    print("Failed to save waveform data.")

def main():
    root = Tk()
    gui = OscilloscopeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()