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

def find_marker(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_b_pink = np.array([20, 120, 170])  # pink
    u_b_pink = np.array([230, 255, 235])

    l_b_green_blue = np.array([20, 0, 0])  # green, blue
    u_b_green_blue = np.array([140, 140, 190])

    l_b_orange = np.array([0, 160, 225])  # orange
    u_b_orange = np.array([255, 255, 255])

    l_b_white = np.array([0, 0, 190])  # white
    u_b_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, l_b_pink, u_b_pink) | cv2.inRange(hsv, l_b_green_blue, u_b_green_blue) | cv2.inRange(hsv,
                                                                                                                 l_b_orange,
                                                                                                                 u_b_orange) | cv2.inRange(
        hsv, l_b_white, u_b_white)
    kernel = np.ones((25, 25), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    contours0, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    cv2.drawContours(image, contours0, -1, 255, 3)
    # print(contours0)
    if len(contours0) == 0:
        return None
    c = max(contours0, key=cv2.contourArea)
    return cv2.boundingRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 25.5
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 7.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length

with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_preview()
    # Camera warm-up time
    sleep(2)

    camera.framerate = 1
    # camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    # loop over the images
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        image = frame.array
        imge = np.copy(image)
        rawCapture.truncate(0) 
        res = find_marker(image)
        if res:
            x, y, w, h = res
            known_width = 7.00
            focal_length = 0.315

            resolution = [640, 480]

            print(f'Top left corner cords: {x}, {y}\nDimentions: {w}, {h}')

            avg_res = (resolution[0] + resolution[1]) / 2
            m = avg_res / focal_length

            x = 640 / (resolution[0] / m)
            width_pixels_in_cm = w / x

            print(width_pixels_in_cm)

            distance_mm = (known_width * focal_length) / width_pixels_in_cm  # [mm*mm /mm = mm]

            print(distance_mm)

            if distance_mm < 20:
                fc.forward(0)
            else:
                fc.forward(3)
        else:
            fc.forward(3)