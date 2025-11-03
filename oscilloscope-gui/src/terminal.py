from tkinter import Text, Scrollbar, Frame

class Terminal(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.create_widgets()

    def create_widgets(self):
        self.text_area = Text(self, wrap='word', height=15, width=50)
        self.text_area.pack(side='left', fill='both', expand=True)

        self.scrollbar = Scrollbar(self, command=self.text_area.yview)
        self.scrollbar.pack(side='right', fill='y')

        self.text_area.config(yscrollcommand=self.scrollbar.set)

        self.pack()

    def write(self, message):
        self.text_area.insert('end', message + '\n')
        self.text_area.see('end')

    def clear(self):
        self.text_area.delete('1.0', 'end')