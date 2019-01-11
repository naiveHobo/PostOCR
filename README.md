# PostOCR
PostOCR is a GUI tool, written using python3 and tkinter, which detects and corrects 
errors that creep in after running an OCR on a PDF document.

## Installation
To install PostOCR along with the dependencies:
```
sudo apt install python3-tk
sudo apt install tesseract-ocr

git clone https://github.com/naiveHobo/PostOCR.git

cd PostOCR/

sudo pip3 install .
```

## Instructions
To start PostOCR:
```
from tkinter import Tk
from PostOCR import PostOCR


root = Tk()
PostOCR()
root.mainloop()
```

## Dependencies

```
python3
tkinter
pdfplumber
PyPDF2
pytesseract
tesseract-ocr
Pillow
```
