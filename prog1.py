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
TIME4TURN = 2.1
TURN_90_rigth = 1.02
TURN_90_left = 1.1
TURN_POWER = 10
TIME4AVOIDANCE = 1
AVOIDANCE_FORWARD_TIME = 2.5


def find_marker(frame):
    frame = frame[300:, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask_l = np.array([10, 70, 158]) #pink
    # mask_h = np.array([190, 180, 220])
    mask_l = np.array([0, 90, 0]) #pink
    mask_h = np.array([111, 217, 148])
    # l_b_green_blue= np.array([20, 0, 0]) # green, blue
    # u_b_green_blue = np.array([140, 140, 190])

    # l_b_orange = np.array([0, 160, 225]) # orange
    # u_b_orange = np.array([255, 255, 255])

    # l_b_white = np.array([0, 0, 190]) # white
    # u_b_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, mask_l, mask_h)# | cv2.inRange(hsv, l_b_green_blue, u_b_green_blue) | cv2.inRange(hsv, l_b_orange, u_b_orange) | cv2.inRange(hsv, l_b_white, u_b_white)
    kernel = np.ones((5, 5),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erodtion = cv2.erode(dilation, kernel, iterations=1)
    cv2.imwrite('out.png', erodtion)

    contours0, hierarchy = cv2.findContours(erodtion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    # print(contours0)
    if len(contours0) == 0:
        # print('no conts')
        return None
    cv2.drawContours(frame, contours0, -1, 255, 3)
    c = max(contours0, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def detect_final_mark(frame):
    frame = frame[300:, :]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask_l = np.array([10, 70, 158]) #pink
    # mask_h = np.array([190, 180, 220])
    mask_l = np.array([148, 78, 105]) #pink
    mask_h = np.array([255, 255, 255])
    # l_b_green_blue= np.array([20, 0, 0]) # green, blue
    # u_b_green_blue = np.array([140, 140, 190])

    # l_b_orange = np.array([0, 160, 225]) # orange
    # u_b_orange = np.array([255, 255, 255])

    # l_b_white = np.array([0, 0, 190]) # white
    # u_b_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, mask_l, mask_h)# | cv2.inRange(hsv, l_b_green_blue, u_b_green_blue) | cv2.inRange(hsv, l_b_orange, u_b_orange) | cv2.inRange(hsv, l_b_white, u_b_white)
    kernel = np.ones((5, 5),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erodtion = cv2.erode(dilation, kernel, iterations=1)
    cv2.imwrite('out.png', erodtion)

    contours0, hierarchy = cv2.findContours(erodtion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    # print(contours0)
    if len(contours0) == 0:
        # print('no conts')
        return None
    cv2.drawContours(frame, contours0, -1, 255, 3)
    c = max(contours0, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

def distance_to_obstacle(img):
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

        return distance_mm

    return 1000
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 25.5
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 7.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
# cv2.namedWindow( "im", cv2.WINDOW_AUTOSIZE )

with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_preview()
    # Camera warm-up time
    sleep(2)

    camera.framerate = 30
    start_time = time.time()
    # camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    state = 0
    fc.forward(POWER)
    time_passed = 0
    pass_state = 0
    i = 0
    # loop over the images
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        image = frame.array
        # cv2.imshow('im',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # imge = np.copy(image)
        rawCapture.truncate(0)
        # dist_to_obs = distance_to_obstacle(image)
        res = find_marker(image)
        # print(res)
        if res:
            x,y,w,h = res
            cv2.rectangle(image, (x, y + 300), (x + w , y + h + 300), (0, 255, 0), 2)
            cv2.imwrite(f'out{i}.png', image)
            i += 1
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
                # fc.turn_left(TURN_POWER)
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
            # TODO detect end goal
            if cur_time - start_time + time_passed >= TIME2TARGET * 0.66:
                mark_pos = detect_final_mark(image)
                if not mark_pos:
                    fc.stop()
                    break
                x, y, w, h = mark_pos
                print(x, y, w, h)
            if cur_time - start_time >= TIME2TARGET * 1.3 - time_passed:
                fc.stop()
                break

                time_passe



