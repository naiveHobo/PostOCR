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

Download the HoboNet model and data files:
- [Model file](https://drive.google.com/open?id=1MydXPvgGsRQ31DTOh45YG0lwRKYFk3xq):  
Place the model.h5 file in **PostOCR/HoboNet/model/** directory.
- [Data file](https://drive.google.com/open?id=1fjwQ7bMx4zYlBt1uTi8pI6Fe1Ej1vL3x):  
Place the data.h5 file in **PostOCR/HoboNet/data/** directory.

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
