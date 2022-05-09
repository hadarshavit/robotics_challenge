# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from time import sleep
import picar_4wd as fc

TIME2TARGET = 7.6
POWER = 3
TIME4TURN = 1.9
TURN_90_rigth = 1.02
TURN_90_left = 1.1
TURN_POWER = 10
TIME4AVOIDANCE = 1
AVOIDANCE_FORWARD_TIME = 2.5


def find_marker(frame):
    frame = frame[300:, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_l = np.array([0, 90, 0]) 
    mask_h = np.array([111, 217, 148])

    mask = cv2.inRange(hsv, mask_l, mask_h)
    kernel = np.ones((5, 5),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erodtion = cv2.erode(dilation, kernel, iterations=1)

    contours0, hierarchy = cv2.findContours(erodtion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    if len(contours0) == 0:
        return None
    cv2.drawContours(frame, contours0, -1, 255, 3)
    c = max(contours0, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def detect_final_mark(frame):
    frame = frame[300:, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_l = np.array([148, 78, 105]) 
    mask_h = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, mask_l, mask_h)
    kernel = np.ones((5, 5),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erodtion = cv2.erode(dilation, kernel, iterations=1)

    contours0, hierarchy = cv2.findContours(erodtion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

    if len(contours0) == 0:
        return None
    cv2.drawContours(frame, contours0, -1, 255, 3)
    c = max(contours0, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def find_obstacle_edges(img):
        img = img[300:, :]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 6)
        img_canny = cv2.Canny(img_blur, 119, 175)
        kernel = np.ones((9, 3))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=7)
        final = cv2.erode(img_dilate, kernel, iterations=7)

        contours0, hierarchy = cv2.findContours(final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

        if len(contours0) == 0:
            return None

        c = max(contours0, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_preview()
    # Camera warm-up time
    sleep(2)

    camera.framerate = 15
    start_time = time.time()
    # camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    state = 0
    fc.forward(POWER)
    time_passed = 0
    pass_state = 0
    final_steering = 0
    i = 0
    # loop over the images
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        image = frame.array
        rawCapture.truncate(0)
        res = find_marker(image)

        if res:
            print('color' ,res)
            res = find_obstacle_edges(image)
            if not res:
                dist_to_obs = 1000
            else:
                x,y,w,h = res
                print('edged', x, y, w, h)

                i += 1
                if x >= 590 or x <= 50:
                    dist_to_obs = 1000
                else:
                    dist_to_obs = distance_to_camera(knownWidth=7, focalLength=530, perWidth=w)
        else:
            dist_to_obs = 10000
        cur_time = time.time()
        print(state, pass_state, time_passed, dist_to_obs)
        if pass_state == 1:
            if cur_time - start_time >= TURN_90_rigth:
                pass_state = 2
                fc.forward(POWER)
                start_time = cur_time
        elif pass_state == 2:
            if cur_time - start_time >= TIME4AVOIDANCE:
                pass_state = 3
                fc.turn_left(TURN_POWER)
                start_time = cur_time
        elif pass_state == 3:
            if cur_time - start_time >= TURN_90_left:
                pass_state = 4
                fc.forward(POWER)
                start_time = cur_time
        elif pass_state == 4:
            if cur_time - start_time >= AVOIDANCE_FORWARD_TIME:
                time_passed += cur_time - start_time
                pass_state = 5
                fc.turn_left(TURN_POWER)
                start_time = cur_time
        elif pass_state == 5:
            if cur_time - start_time >= TURN_90_left:
                pass_state = 6
                fc.forward(POWER)
                start_time = cur_time
        elif pass_state == 6:
            if cur_time - start_time >= TIME4AVOIDANCE:
                pass_state = 7
                fc.turn_right(TURN_POWER)
                start_time = cur_time
        elif pass_state == 7:
            if cur_time - start_time >= TURN_90_rigth:
                pass_state = 0
                fc.forward(POWER)
                start_time = cur_time
        elif state == 0:
            if dist_to_obs < 25:
                pass_state = 1
                time_passed += cur_time - start_time
                fc.turn_right(TURN_POWER)
                start_time = cur_time
            if cur_time - start_time >= TIME2TARGET - time_passed:
                state = 1
                time_passed = 0
                fc.turn_right(TURN_POWER)
                start_time = cur_time
        elif state == 1:
            if cur_time - start_time >= TIME4TURN:
                state = 2
                time_passed = 0
                fc.forward(POWER)
                start_time = cur_time
        elif state == 2:
            if dist_to_obs < 25:
                pass_state = 1
                time_passed += cur_time - start_time
                fc.turn_right(TURN_POWER)
                start_time = cur_time

            if cur_time - start_time + time_passed >= TIME2TARGET * 0.66:
                mark_pos = detect_final_mark(image)
                print('mark pos', mark_pos)
                if not mark_pos:
                    print('no mark, going forward')
                    fc.forward(POWER)
                    state = 3
                    start_time = cur_time
                else:
                    x, y, w, h = mark_pos
                    print('fina mark', x, y, w, h)
                    mid = x + w / 2
                    if mid < 200 and w < 80: # turn left
                        final_steering = 1
                        fc.turn_left(1)
                    elif mid > 440 and w < 80: # turn right:
                        final_steering = 2 
                        fc.turn_right(1)
                    else:
                        fc.forward(POWER)

        elif state == 3:
            fc.forward(POWER)
            if cur_time - start_time >= 1.5:
                fc.stop()
                break



