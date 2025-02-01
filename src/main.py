import cv2 as cv
import drawLine

def testLineDraw():
    img = cv.imread('test_images\\testFootballImg.jpg')
    cv.imshow("Display", img)
    cv.waitKey(0)
    img = drawLine.drawLine(img, 100, 600, 2500, 600)
    cv.imshow("Display", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    testLineDraw()


if __name__ == '__main__':
    main()

