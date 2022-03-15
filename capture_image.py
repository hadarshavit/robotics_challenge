
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep

if __name__ == '__main__':
    with PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        # Camera warm-up time
        sleep(2)
        camera.capture('sample_imgs/p4.jpg')