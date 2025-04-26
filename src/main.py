import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np
import time


from multiprocessing.pool import ThreadPool
from collections import deque


def nothing(x):
    pass

def warpBack(warped_line, inverse_homog, target_shape = [1920, 1080]):
    unwarped = cv.warpPerspective(warped_line, inverse_homog, (target_shape[0], target_shape[1]))
    return unwarped


def drawLine(warped, yardline = 50):
    h, w, c = warped.shape
    yard_percentage = yardline / 100
    playable_field = w * (98 / 120)  # 98/120 of the field is a yard line
    endzone_length = w * (11 / 120)  # first viable yard line is 11 yards from the end zone
    width_marker = int(endzone_length + (yard_percentage * playable_field))
    line_img = warped.copy()

    line_width = 3
    start_x = width_marker - line_width
    end_x = width_marker + line_width


    hsv = cv.cvtColor(line_img, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(hsv, (60, 70, 0), (100, 255, 170)) # data set
    mask = cv.inRange(hsv, (50, 50, 0), (100, 255, 200))  # room
    imask = mask > 0
    yellow = np.zeros_like(line_img, np.uint8)
    yellow[:, :, 1:] = 255
    line_img[imask] = yellow[imask]

    line = np.zeros_like(line_img, np.uint8)
    line[:, start_x:end_x] = line_img[:, start_x:end_x]

    warped_line = warped.copy()
    alpha = 0.5
    warped_line = np.where(line > 0, cv.addWeighted(warped_line, alpha, line, 1 - alpha, 0), warped_line)

    # cv.line(line_img, (width_marker, 0), (width_marker, h), (0, 255, 255), 5)
    # warped_line = cv.addWeighted(warped, alpha, line_img, 1 - alpha, 0)
    return warped_line

# Warps the field to a rectangle
# Returns the resulting warped image
def warpField(frame, homog, target_shape = [900, 400]):
    warped_image = cv.warpPerspective(frame, homog, (target_shape[0], target_shape[1]))
    return warped_image

# Calculates the homography to warp from the original image to the rectangular field
# Corners: The corners of the field in format ((tl), (tr), (bl), (br))
# Returns the homography and it's inverse.
def calculateHomography(corners):
    # Football field is 360ft by 160ft with a ratio of 2.25:1
    target_shape = [900, 400] # h = 400, w = 900; 900/400 = 2.25
    src = np.array([corners[0][1], corners[0][0], # tl
                    corners[1][1], corners[1][0], # tr
                    corners[3][1], corners[3][0], # br
                    corners[2][1], corners[2][0]  # bl
                    ]).reshape((4, 2))
    dst = np.array([0, 0, # tl
                    target_shape[0], 0, # tr
                    target_shape[0], target_shape[1], # br
                    0, target_shape[1] # bl
                    ]).reshape((4, 2))

    homog = cv.findHomography(src, dst)[0]
    inverse = cv.findHomography(dst, src)[0]
    return homog, inverse

# Returns the location of the 4 corners of the field.
# Frame: image of the field
def getCorners(frame):
    # cv.imshow('Original', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    start_time = time.time()

    # find the green pixels in the image
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(hsv, (50, 70, 0), (100, 255, 170)) # data set green
    mask = cv.inRange(hsv, (50, 50, 0), (100, 255, 200))  # room green
    # mask = cv.inRange(hsv, (50, 50, 0), (100, 255, 255))  # White and green I guess
    imask = mask > 0
    green = np.zeros_like(frame, np.uint8)
    green[imask] = frame[imask]

    green_time = time.time()

    # Erode to get rid of any misc green things in frame.
    erode_kernel_size = 5
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8) / (erode_kernel_size ** 2)
    green = cv.erode(green, erode_kernel, iterations=1)

    erode1_time = time.time()

    # Dilate to fill any holes the came from the center of the field
    dilate_kernel_size = 7
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8) / (dilate_kernel_size ** 2)
    green = cv.dilate(green, dilate_kernel, iterations=5)

    dilate_time = time.time()

    # cv.imshow('Dilated Green', green)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Erode back down to just the field
    green = cv.erode(green, dilate_kernel, iterations=8)

    erode2_time = time.time()

    # cv.imshow('Erode 2', green)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # dilate_threshold = [0,0,0]
    # dilate_mask = (green > dilate_threshold).any(axis=2)
    # green[dilate_mask] = frame[dilate_mask]
    green = np.where(green > 0, frame, 0)

    mask_time = time.time()

    # cv.imshow('Thresholded Green', green)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    thresh = 0.03
    gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)

    dst = cv.cornerHarris(gray, 2, 15, 0.04)

    # dst = cv.dilate(dst, None)

    threshold_value = thresh * np.max(dst)
    corners = np.argwhere(dst > threshold_value)

    if corners.size == 0:
        return None, corners

    # # temp to see if working
    # temp = img.copy()
    # for corner in corners:
    #     x = corner[0]
    #     y = corner[1]
    #     temp[x, y] = [0, 0, 255]
    # cv.imshow('Corners', temp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    h, w, c = frame.shape

    normal_metric = corners[:, 0] + corners[:, 1]

    top_left_index = normal_metric.argmin()
    top_left = corners[top_left_index]

    bottom_right_index = normal_metric.argmax()
    bottom_right = corners[bottom_right_index]

    inverse_metric = (h - corners[:, 0]) + corners[:, 1]

    bottom_left_index = inverse_metric.argmin()
    bottom_left = corners[bottom_left_index]

    top_right_index = inverse_metric.argmax()
    top_right = corners[top_right_index]

    # # temp to see if working
    # img[top_left[0], top_left[1]] = [0, 0, 255]
    # img[top_right[0], top_right[1]] = [0, 0, 255]
    # img[bottom_right[0], bottom_right[1]] = [0, 0, 255]
    # img[bottom_left[0], bottom_left[1]] = [0, 0, 255]
    # cv.imshow('Corners', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    corner_time = time.time()

    print("Green Time: " + str(green_time-start_time) + ". Erode 1 Time: " + str(erode1_time-green_time) +
          ". Dilate Time: " + str(dilate_time-erode1_time) + ". Erode 2 Time: " + str(erode2_time-dilate_time) +
          ". Mask Time: " + str(mask_time-erode2_time) + ". Corner Time: " + str(corner_time-mask_time) +
          ". Total Time: " + str(corner_time-start_time) + ".")

    # If any of the corners match we didn't find enough corners to warp
    if np.array_equal(top_left, top_right) or np.array_equal(top_left, bottom_left) or np.array_equal(top_left,
                                                                                                      bottom_right):
        return None, corners
    if np.array_equal(top_right, bottom_left) or np.array_equal(top_right, bottom_right):
        return None, corners
    if np.array_equal(bottom_left, bottom_right):
        return None, corners

    return [top_left, top_right, bottom_left, bottom_right], corners

# Processes and individual frame
# Frame: A image that is to be processed
def processFrame(frame):
    # Timer for testing
    start_time = time.time()

    # get corners
    corners, _ = getCorners(frame)
    if corners == None: # Didn't find enough corners
        return frame
    corner_time = time.time()

    # calculate homography and inverse homography
    homog, inverse_homog = calculateHomography(corners)

    # warp image
    warped = warpField(frame, homog)
    homog_time = time.time()

    # draw line
    yardline = cv.getTrackbarPos('Yard Line', 'Yard Line')
    warped_line = drawLine(warped, yardline)
    line_time = time.time()

    # warp back
    unwarped = warpBack(warped_line, inverse_homog)

    # final image
    final = np.where(unwarped > 0, unwarped, frame)

    # Timer for testing
    end_time = time.time()
    print("Corner Time: " + str(corner_time-start_time) + ". Homog Time: " + str(homog_time-corner_time) +
          ". Line Time: " + str(line_time-homog_time) + ". Unwarped Time: " + str(end_time-line_time) +
          ". Total Time: " + str(end_time-start_time))


    # test print of things
    # cv.imshow('Display', final)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return final

def startVideo():
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    prev_time = 0
    while True:
        ret, frame = capture.read()
        processed_frame = processFrame(frame)

        # Temp framerate
        new_time = time.time()
        fps = str(int(1 / (new_time - prev_time)))
        prev_time = new_time
        cv.putText(processed_frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)

        cv.imshow('Display', processed_frame)

        input_key = cv.waitKey(1)
        if input_key == ord('q') or input_key == 27:
            break
    capture.release()
    cv.destroyAllWindows()

def startVideoParallel():
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    cv.namedWindow("Yard Line")
    cv.resizeWindow("Yard Line", 500, 45)
    cv.createTrackbar("Yard Line", "Yard Line", 0, 100, nothing)

    # prev_time = 0
    while True:
        while len(pending) > 0 and pending[0].ready(): # pop off of queue
            processed_frame = pending.popleft().get()
            cv.imshow('Display', processed_frame)
        # add frames to queue
        if len(pending) < threadn:
            ret, frame = capture.read()

            # Temp framerate
            # new_time = time.time()
            # fps = str(int(1 / (new_time - prev_time)))
            # prev_time = new_time

            task = pool.apply_async(processFrame, (frame,))
            pending.append(task)

        input_key = cv.waitKey(1)
        if input_key == ord('q') or input_key == 27:
            break
    capture.release()
    cv.destroyAllWindows()


def main():
    # startVideo()
    startVideoParallel()
    # img = cv.imread('test_images/1.jpg')
    # processFrame(img)
    # directory_path = 'test_images'
    # for filename in os.listdir(directory_path):
    #     filename = os.path.join(directory_path, filename)
    #     img = cv.imread(filename)
    #     processFrame(img)



if __name__ == '__main__':
    main()

