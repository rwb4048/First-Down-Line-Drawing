import cv2 as cv
import sys

def drawLine(img, x1, y1, x2, y2):
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
    return img