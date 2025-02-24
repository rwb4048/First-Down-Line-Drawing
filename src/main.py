import cv2 as cv
import numpy as np
from matplotlib.pyplot import imshow

import drawLine

def testLineDraw():
    # img = cv.imread('test_images\\testFootballImg.jpg')
    img = cv.imread('test_images\\TestImg2.jpg')
    cv.imshow("Display", img)
    cv.waitKey(0)
    img = drawLine.drawLine(img, 237, 20, 237, 290)
    cv.imshow("Display", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def testCorners():
    img = cv.imread('test_images\\IMG_0485.jpeg')
    rows, cols, _channels = map(int, img.shape)
    img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
    img = cv.pyrDown(img, dstsize=(cols // 4, rows // 4))
    cv.imshow("Display", img)
    cv.waitKey(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Display", gray)
    cv.waitKey(0)

    gray = np.float32(gray)

    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.2 * dst.max()] = [0, 0, 255]

    cv.imshow('Display', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def testHomography():
    img = cv.imread('test_images\\IMG_0485.jpeg')
    rows, cols, _channels = map(int, img.shape)
    img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
    img = cv.pyrDown(img, dstsize=(cols // 4, rows // 4))

    src = np.array([154, 244,
                    944, 479,
                    76, 478,
                    869, 244,

                    ]).reshape((4, 2))

    dst = np.array([0, 50,
                    1000, 500,
                    0, 500,
                    1000, 50,
                    ]).reshape((4, 2))

    img = cv.transform(img, src, dst)
    # tform = cv.transform.estimate_transform('projective', src, dst)

    cv.imshow("Display", img)
    cv.waitKey(0)

def test2():
    def point_capture(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Coordinate of Image: ({x}, {y})")
            cv.putText(img, f'({x}, {y})', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('Display', img)
    img = cv.imread('test_images\\IMG_0485.jpeg')
    rows, cols, _channels = map(int, img.shape)
    img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
    img = cv.pyrDown(img, dstsize=(cols // 4, rows // 4))
    cv.imshow("Display", img)
    cv.setMouseCallback("Display", point_capture)
    cv.waitKey(0)


def main():
    test2()


if __name__ == '__main__':
    main()

