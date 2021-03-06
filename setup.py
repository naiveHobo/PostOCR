import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(name='PostOCR',
                 version='0.1',
                 description='Tkinter based GUI for detection and correction of OCR errors',
                 url='https://github.com/naiveHobo/PostOCR.git',
                 author='naiveHobo',
                 author_email='sarthakmittal2608@gmail.com',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'Pillow',
                     'pdfplumber',
                     'PyPDF2',
                     'pytesseract',
                     'tensorflow-gpu'
                 ],
                 zip_safe=False)
