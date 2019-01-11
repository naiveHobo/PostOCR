from tkinter import *

from .config import BACKGROUND_COLOR


class OCRReader(Frame):

    def __init__(self, master, text, label_text, show_errors, **kw):
        Frame.__init__(self, master, **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)

        Label(self, text=label_text, anchor='nw', width=70,
              font="OpenSans 22 bold", fg='white', bg=BACKGROUND_COLOR, bd=2).grid(row=0, column=0, padx=20, pady=20)

        text_frame = Frame(self, height=440, width=550, bg=BACKGROUND_COLOR, bd=2, relief=SUNKEN)
        text_frame.grid(row=1, column=0)

        text_frame.grid_propagate(False)

        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        self.text_box = Text(text_frame, borderwidth=3, relief="sunken", bg=BACKGROUND_COLOR,
                             fg='white', font="OpenSans 12", wrap='word')
        self.text_box.tag_configure("correction", foreground='red')

        if not show_errors:
            self.text_box.insert('1.0', text)
        else:
            splits = text.split('<CORRECTION>')
            splits = [split for split in splits if split]
            if '</CORRECTION>' not in splits[0]:
                self.text_box.insert('1.0', splits[0])
            else:
                words = splits[0].split('</CORRECTION>')
                self.text_box.insert('1.0', words[0], 'correction')
                self.text_box.insert('end', words[1])
            for split in splits[1:]:
                if '</CORRECTION>' not in split:
                    self.text_box.insert('end', split)
                else:
                    words = split.split('</CORRECTION>')
                    self.text_box.insert('end', words[0], 'correction')
                    self.text_box.insert('end', words[1])

        self.text_box.config(state=DISABLED)
        self.text_box.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        scroll_bar = Scrollbar(text_frame, command=self.text_box.yview, bg=BACKGROUND_COLOR)
        scroll_bar.grid(row=0, column=1, sticky='nsew')

        self.text_box['yscrollcommand'] = scroll_bar.set
