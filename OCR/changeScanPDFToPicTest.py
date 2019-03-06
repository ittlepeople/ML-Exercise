#! python
'''
This demo extracts all images of a PDF as PNG files that are referenced
by pages.
Runtime is determined by number of pages and volume of stored images.
Usage:
------
extract_img1.py input.pdf
Changes:
--------
(1)
Do not use pix.colorspace when checking for CMYK images, because it may be None
if the image is a stencil mask. Instead, we now check pix.n - pix.alpha < 4
to confirm that an image is not CMYK.
IAW: the repr() of a stencil mask pixmap looks like
"Pixmap(None, fitz.IRect(0, 0, width, height), 1)", and we have
width * height == len(pix.samples).
(2)
Pages may reference the same image multiple times. To avoid to also extracting
duplicate images, we maintain a list of xref numbers.
'''
from __future__ import print_function
import fitz
import sys, time, os
from  PIL import  Image
import pytesseract


def pdfOcr(pdfFileName):
    t0 = time.clock()

    # 输入pdf文件名称 以及 输出图片目录
    imgdir = "result"  # found images are stored here
    # pdfFileName = '../data/美军作战.pdf'
    # f = open('out.txt', 'w')
    # 清空文件夹
    for i in os.listdir(imgdir):
       path_file = os.path.join(imgdir, i)  # 取文件路径
       if os.path.isfile(path_file):
          os.remove(path_file)

    # 打开文件
    doc = fitz.open(pdfFileName)  # the PDF  './new/opencv.pdf'    './data/1.pdf'
    print('pageCount:', doc.pageCount)

    AllText = ''

    # 先判断Pdf是否是文字类型
    for i in range(len(doc)):
        page = doc[i]
        text = page.getText("text")
        AllText += text

    AllText = AllText.strip().replace(' ', '')
    print('AllText:', len(AllText), AllText)

    if len(AllText) == 0:
        # 这是扫描版pdf需要先提取图片，然后OCR
        # 对每页提取图片
        for i in range(doc.pageCount):
            page = doc[i]
            pix = page.getPixmap()

            # print('pix:', pix)
            print('save: ', os.path.join(imgdir, "p%i.png" % (i)))
            pix.writePNG(os.path.join(imgdir, "p%i.png" % (i)))

            # 这里实现pdf转化文字的部分
            img_path = os.path.join(imgdir, "p%i.png" % (i))
            text = pytesseract.image_to_string(Image.open(img_path), lang='chi_sim')
            text = text.replace(' ', '').strip()
            print(text)
            AllText += text
        # f.write(text)
        # f.close()
    else:
        print('out txt:', AllText)

    t1 = time.clock()
    print("run time", round(t1 - t0, 2))
    return AllText


if __name__ == '__main__':
    pdfFileName = '../data/OCRtxt.pdf'
    text = pdfOcr(pdfFileName)
