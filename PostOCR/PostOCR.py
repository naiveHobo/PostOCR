import os
import json
import h5py
import subprocess
import pdfplumber
import pytesseract
from PIL import Image
from wand.exceptions import WandException

from tkinter import *
from tkinter import filedialog, simpledialog, messagebox

import keras
import numpy as np

from .config import ROOT_PATH, BACKGROUND_COLOR, HIGHLIGHT_COLOR
from .hoverbutton import HoverButton
from .helpbox import HelpBox
from .menubox import MenuBox
from .display_canvas import DisplayCanvas
from .ocr_reader import OCRReader
from .model import get_model


class PostOCR(Frame):

    def __init__(self, master=None, **kw):
        Frame.__init__(self, master, **kw)
        self.pdf = None
        self.page = None
        self.path = None
        self.total_pages = 0
        self.pageidx = 0
        self.scale = 1.0
        self.rotate = 0
        self.save_path = None
        self.hobonet = None
        self.vocab = None
        self.ocr_text = None
        self.ocr_text_path = None
        self._load_hobonet()
        self._init_ui()

    def _load_hobonet(self, path=os.path.join(ROOT_PATH, 'HoboNet/')):
        hf = h5py.File(os.path.join(path, 'data/data.h5'), 'r')
        self.vocab = json.loads(hf['vocab'].value)
        self.hobonet = get_model()
        self.hobonet.load_weights(os.path.join(path, 'model/model.h5'))

    def _init_ui(self):
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        h = hs - 100
        w = int(h / 1.414) + 100
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.master.title("PostOCR")

        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(0, weight=0)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)

        self.configure(bg=BACKGROUND_COLOR, bd=0)

        tool_frame = Frame(self, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)
        pdf_frame = Frame(self, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)

        tool_frame.grid(row=0, column=0, sticky='news')
        pdf_frame.grid(row=0, column=1, sticky='news')

        # Tool Frame
        tool_frame.columnconfigure(0, weight=1)
        tool_frame.rowconfigure(0, weight=0)
        tool_frame.rowconfigure(1, weight=1)
        tool_frame.rowconfigure(2, weight=0)
        tool_frame.rowconfigure(3, weight=2)

        options = MenuBox(tool_frame, image_path=os.path.join(ROOT_PATH, 'widgets/options.png'))
        options.grid(row=0, column=0)

        options.add_item('Open File...', self._open_file, seperator=True)
        options.add_item('Search...', self._search_text, seperator=True)
        options.add_item('Run OCR', self._run_ocr)
        options.add_item('Load HoboNet...', self._load_hobonet_popup)
        options.add_item('Find OCR Errors', self._detect_errors)
        options.add_item('Fix OCR Errors', self._correct_errors, seperator=True)
        options.add_item('Help...', self._help, seperator=True)
        options.add_item('Exit', self.master.quit)

        tools = Frame(tool_frame, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)
        tools.grid(row=2, column=0)

        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/open_file.png'), command=self._open_file,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Open File",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/clear.png'), command=self._clear,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Clear",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/search.png'), command=self._search_text,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Search Text",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/extract.png'), command=self._extract_text,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Extract Text", keep_pressed=True,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/ocr.png'), command=self._run_ocr,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Run OCR",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/find.png'), command=self._detect_errors,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Find OCR Errors",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)
        HoverButton(tools, image_path=os.path.join(ROOT_PATH, 'widgets/fix.png'), command=self._correct_errors,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Fix OCR Errors",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(pady=2)

        HoverButton(tool_frame, image_path=os.path.join(ROOT_PATH, 'widgets/help.png'), command=self._help,
                    width=50, height=50, bg=BACKGROUND_COLOR, bd=0, tool_tip="Help",
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).grid(row=3, column=0, sticky='s')

        # PDF Frame
        pdf_frame.columnconfigure(0, weight=1)
        pdf_frame.rowconfigure(0, weight=0)
        pdf_frame.rowconfigure(1, weight=0)

        page_tools = Frame(pdf_frame, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)
        page_tools.grid(row=0, column=0, sticky='news')

        page_tools.rowconfigure(0, weight=1)
        page_tools.columnconfigure(0, weight=1)
        page_tools.columnconfigure(1, weight=0)
        page_tools.columnconfigure(2, weight=2)
        page_tools.columnconfigure(3, weight=0)
        page_tools.columnconfigure(4, weight=1)

        nav_frame = Frame(page_tools, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)
        nav_frame.grid(row=0, column=1, sticky='ns')

        HoverButton(nav_frame, image_path=os.path.join(ROOT_PATH, 'widgets/first.png'),
                    command=self._first_page, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=LEFT, expand=True)
        HoverButton(nav_frame, image_path=os.path.join(ROOT_PATH, 'widgets/prev.png'),
                    command=self._prev_page, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=LEFT, expand=True)

        self.page_label = Label(nav_frame, bg=BACKGROUND_COLOR, bd=0, fg='white', font='Arial 8',
                                text="Page {} of {}".format(self.pageidx, self.total_pages))
        self.page_label.pack(side=LEFT, expand=True)

        HoverButton(nav_frame, image_path=os.path.join(ROOT_PATH, 'widgets/next.png'),
                    command=self._next_page, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=LEFT, expand=True)
        HoverButton(nav_frame, image_path=os.path.join(ROOT_PATH, 'widgets/last.png'),
                    command=self._last_page, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=LEFT, expand=True)

        zoom_frame = Frame(page_tools, bg=BACKGROUND_COLOR, bd=0, relief=SUNKEN)
        zoom_frame.grid(row=0, column=3, sticky='ns')

        HoverButton(zoom_frame, image_path=os.path.join(ROOT_PATH, 'widgets/rotate.png'),
                    command=self._rotate, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=RIGHT, expand=True)
        HoverButton(zoom_frame, image_path=os.path.join(ROOT_PATH, 'widgets/fullscreen.png'),
                    command=self._fit_to_screen, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=RIGHT, expand=True)

        self.zoom_label = Label(zoom_frame, bg=BACKGROUND_COLOR, bd=0, fg='white', font='Arial 8',
                                text="Zoom {}%".format(int(self.scale * 100)))
        self.zoom_label.pack(side=RIGHT, expand=True)

        HoverButton(zoom_frame, image_path=os.path.join(ROOT_PATH, 'widgets/zoomout.png'),
                    command=self._zoom_out, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=RIGHT, expand=True)
        HoverButton(zoom_frame, image_path=os.path.join(ROOT_PATH, 'widgets/zoomin.png'),
                    command=self._zoom_in, bg=BACKGROUND_COLOR, bd=0,
                    highlightthickness=0, activebackground=HIGHLIGHT_COLOR).pack(side=RIGHT, expand=True)

        canvas_frame = Frame(pdf_frame, bg=BACKGROUND_COLOR, bd=1, relief=SUNKEN)
        canvas_frame.grid(row=1, column=0, sticky='news')

        self.canvas = DisplayCanvas(canvas_frame, page_height=h-42, page_width=w-70)
        self.canvas.pack()

        self.grid(row=0, column=0, sticky='news')

        self.master.minsize(height=h, width=w)
        self.master.maxsize(height=h, width=w)

    def _reset(self):
        self.canvas.clear()
        self.pdf = None
        self.page = None
        self.total_pages = 0
        self.pageidx = 0
        self.scale = 1.0
        self.rotate = 0
        self.page_label.configure(text="Page {} of {}".format(self.pageidx, self.total_pages))
        self.zoom_label.configure(text="Zoom {}%".format(int(self.scale * 100)))
        self.master.title("PostOCR")

    def _clear(self):
        if self.pdf is None:
            return
        self.canvas.reset()
        self._update_page()

    def _zoom_in(self):
        if self.pdf is None:
            return
        if self.scale == 2.5:
            return
        self.scale += 0.1
        self._update_page()

    def _zoom_out(self):
        if self.pdf is None:
            return
        if self.scale == 0.1:
            return
        self.scale -= 0.1
        self._update_page()

    def _fit_to_screen(self):
        if self.pdf is None:
            return
        if self.scale == 1.0:
            return
        self.scale = 1.0
        self._update_page()

    def _rotate(self):
        if self.pdf is None:
            return
        self.rotate = (self.rotate - 90) % 360
        self._update_page()

    def _next_page(self):
        if self.pdf is None:
            return
        if self.pageidx == self.total_pages:
            return
        self.pageidx += 1
        self._update_page()

    def _prev_page(self):
        if self.pdf is None:
            return
        if self.pageidx == 1:
            return
        self.pageidx -= 1
        self._update_page()

    def _last_page(self):
        if self.pdf is None:
            return
        if self.pageidx == self.total_pages:
            return
        self.pageidx = self.total_pages
        self._update_page()

    def _first_page(self):
        if self.pdf is None:
            return
        if self.pageidx == 1:
            return
        self.pageidx = 1
        self._update_page()

    def _update_page(self):
        page = self.pdf.pages[self.pageidx - 1]
        self.page = page.to_image(resolution=int(self.scale * 80))
        image = self.page.original.rotate(self.rotate)
        self.canvas.update_image(image)
        self.page_label.configure(text="Page {} of {}".format(self.pageidx, self.total_pages))
        self.zoom_label.configure(text="Zoom {}%".format(int(self.scale * 100)))

    def _search_text(self):
        if self.pdf is None:
            return
        text = simpledialog.askstring('Search Text', 'Enter text to search:')
        if text == '' or text is None:
            return
        page = self.pdf.pages[self.pageidx - 1]
        image = page.to_image(resolution=int(self.scale * 80))
        words = [w for w in page.extract_words() if text.lower() in w['text'].lower()]
        image.draw_rects(words)
        image = image.annotated.rotate(self.rotate)
        self.canvas.update_image(image)

    def _extract_text(self):
        if self.pdf is None:
            return
        if not self.canvas.draw:
            self.canvas.draw = True
            self.canvas.configure(cursor='cross')
            return
        self.canvas.draw = False
        self.canvas.configure(cursor='')
        rect = self.canvas.get_rect()
        if rect is None:
            return
        self._clear()
        rect = self._reproject_bbox(rect)
        page = self.pdf.pages[self.pageidx - 1]
        words = page.extract_words()
        min_x = 1000000
        r = None
        for word in words:
            diff = abs(float(word['x0'] - rect[0])) + abs(float(word['top'] - rect[1])) \
                   + abs(float(word['x1'] - rect[2])) + abs(float(word['bottom'] - rect[3]))
            if diff < min_x:
                min_x = diff
                r = word
        image = page.to_image(resolution=int(self.scale * 80))
        image.draw_rect(r)
        image = image.annotated.rotate(self.rotate)
        self.canvas.update_image(image)
        simpledialog.askstring("Extract Text", "Text Extracted:", initialvalue=r['text'])

    def _reproject_bbox(self, bbox):
        bbox = [self.page.decimalize(x) for x in bbox]
        x0, y0, x1, y1 = bbox
        px0, py0 = self.page.page.bbox[:2]
        rx0, ry0 = self.page.root.bbox[:2]
        _x0 = (x0 / self.page.scale) - rx0 + px0
        _y0 = (y0 / self.page.scale) - ry0 + py0
        _x1 = (x1 / self.page.scale) - rx0 + px0
        _y1 = (y1 / self.page.scale) - ry0 + py0
        return [_x0, _y0, _x1, _y1]

    def _run_ocr(self):
        if self.pdf is None:
            return

        text = ''
        for page in self.pdf.pages:
            image = page.to_image(resolution=int(self.scale * 80))
            text += ' ' + pytesseract.image_to_string(image.original)

        dirname = os.path.dirname(self.path)
        filename = os.path.basename(self.path).replace('pdf', 'txt')

        path = filedialog.asksaveasfilename(title='Save OCR text as', defaultextension='.txt',
                                            initialdir=dirname, initialfile=filename,
                                            filetypes=[('Text files', '*.txt'),
                                                       ('PDF files', '*.pdf'),
                                                       ('all files', '.*')])
        if path == '' or path is None:
            return

        self.ocr_text = text.strip()
        self.ocr_text_path = path

        with open(self.ocr_text_path, 'w') as out:
            out.write(self.ocr_text)

        self._display_ocr_text()

    def _display_ocr_text(self):
        if self.pdf is None:
            return
        if not self.ocr_text:
            return
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w, h = 600, 600
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        text_frame = Toplevel(self)
        text_frame.title("OCR Text")
        text_frame.configure(width=w, height=h, bg=BACKGROUND_COLOR, relief=SUNKEN)
        text_frame.geometry('%dx%d+%d+%d' % (w, h, x, y))
        text_frame.minsize(height=h, width=w)
        text_frame.maxsize(height=h, width=w)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        OCRReader(text_frame, text=self.ocr_text, width=w, height=h, bg=BACKGROUND_COLOR, relief=SUNKEN).grid(row=0, column=0)

    @staticmethod
    def _image_to_pdf(path):
        image = Image.open(path)
        pdf = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

        filename = '.'.join(os.path.basename(path).split('.')[:-1]) + '.pdf'
        dirname = os.path.dirname(path)

        path = filedialog.asksaveasfilename(title='Save Converted PDF As', defaultextension='.pdf',
                                            initialdir=dirname, initialfile=filename,
                                            filetypes=[('PDF files', '*.pdf'), ('all files', '.*')])
        if path == '' or path is None:
            return
        with open(path, 'wb') as out:
            out.write(pdf)
        return path

    def _load_hobonet_popup(self):
        path = filedialog.askopenfilename(filetypes=[('HDF5 files', '*.hdf5', '*.h5')],
                                          initialdir=os.getcwd(),
                                          title="Select model file")

        if not path or path == '' or os.path.basename(path).split('.')[-1].lower() not in ['h5', 'hdf5']:
            return

        self.hobonet = keras.models.load_model(path)

    def _detect_errors(self):
        if self.pdf is None:
            return

        if not self.hobonet:
            self._load_hobonet_popup()

    def _correct_errors(self):
        if self.pdf is None:
            return

        if not self.hobonet:
            self._load_hobonet_popup()

        sentences = []
        for i in range(0, len(self.ocr_text), 35):
            sentences.append(self.ocr_text[i:i + 35].lower())

        ret = ""
        for sent in sentences:
            chars = []

            for c in sent:
                if c in self.vocab['char2idx']:
                    chars.append(self.vocab['char2idx'][c])
                else:
                    chars.append(self.vocab['char2idx']['#'])

            m_input = [np.zeros((1, 35)), np.zeros((1, 35))]
            for i, c in enumerate(chars):
                m_input[0][0, i] = c

            for c_i in range(1, 35):
                out = self.hobonet.predict(m_input)
                out_c_i = out[0][c_i - 1].argmax()

                if out_c_i == 0:
                    continue

                ret += self.vocab['idx2char'][str(out_c_i)]
                m_input[1][0, c_i] = out_c_i

        self.ocr_text = ret
        with open(self.ocr_text_path, 'w') as out:
            out.write(self.ocr_text)

        self._display_ocr_text()

    def _load_file(self):
        self._clear()
        path = self.path
        filename = os.path.basename(path)
        if filename.split('.')[-1].lower() in ['jpg', 'png']:
            path = self._image_to_pdf(path)
        try:
            self.pdf = pdfplumber.open(path)
            self.total_pages = len(self.pdf.pages)
            self.pageidx = 1
            self.scale = 1.0
            self.rotate = 0
            self._update_page()
            self.master.title("PostOCR : {}".format(path))
        except WandException:
            res = messagebox.askokcancel("Error", "ImageMagick Policy Error! Should PostOCR try fixing the error?")
            if res == 1:
                self._fix_policy_error()
                messagebox.showinfo("Policy Fixed!", "ImageMagick Policy Error fixed! Restart PostOCR.")
            else:
                messagebox.showerror('Error', 'Could not open file!')
        except (IndexError, IOError, TypeError):
            messagebox.showerror('Error', 'Could not open file!')

    def _open_file(self):
        path = filedialog.askopenfilename(filetypes=[('PDF files', '*.pdf'),
                                                     ('JPG files', '*.jpg'),
                                                     ('PNG files', '*.png'),
                                                     ('all files', '.*')],
                                          initialdir=os.getcwd(),
                                          title="Select files")
        if not path or path == '' or os.path.basename(path).split('.')[-1].lower() not in ['pdf', 'jpg', 'png']:
            return
        self.path = path
        self._load_file()

    def _help(self):
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w, h = 600, 600
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        help_frame = Toplevel(self)
        help_frame.title("Help")
        help_frame.configure(width=w, height=h, bg=BACKGROUND_COLOR, relief=SUNKEN)
        help_frame.geometry('%dx%d+%d+%d' % (w, h, x, y))
        help_frame.minsize(height=h, width=w)
        help_frame.maxsize(height=h, width=w)
        help_frame.rowconfigure(0, weight=1)
        help_frame.columnconfigure(0, weight=1)
        HelpBox(help_frame, width=w, height=h, bg=BACKGROUND_COLOR, relief=SUNKEN).grid(row=0, column=0)

    @staticmethod
    def _fix_policy_error():
        policy_path = "/etc/ImageMagick-6/policy.xml"
        if not os.path.isfile(policy_path):
            policy_path = "/etc/ImageMagick/policy.xml"
        with open(policy_path, 'r') as policy_file:
            data = policy_file.readlines()
            new_data = []

            for line in data:
                if 'MVG' in line:
                    line = '<!-- ' + line + ' -->'
                elif 'PDF' in line:
                    line = '  <policy domain="coder" rights="read|write" pattern="PDF" />\n'
                elif '</policymap>' in line:
                    new_data.append('  <policy domain="coder" rights="read|write" pattern="LABEL" />\n')
                new_data.append(line)

            with open('policy.xml', 'w') as new_file:
                new_file.writelines(new_data)

            subprocess.call(["sudo", "mv", "new_policy.xml", policy_path])
