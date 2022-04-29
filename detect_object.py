import cv2
import numpy as np

def nothing(x):
    pass

# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
# #
# while True:
#     frame = cv2.imread('sample_imgs/p4.jpg')

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     l_h = cv2.getTrackbarPos("LH", "Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")

#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")

#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, l_b, u_b)

#     res = cv2.bitwise_and(frame, frame, mask=mask)

#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("res", res)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cv2.destroyAllWindows()


# cv2.imshow("mask", dilation)
# cv2.waitKey(0)
# final_mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
# final_mask = cv2.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))

# res = cv2.bitwise_and(frame, frame, mask=final_mask)


# print(contours0)
# final_contours = []
# for contour in contours0:
#     area = cv2.contourArea(contour)
#     if area > 2000:
#         final_contours.append(contour)

# for i in range(len(final_contours)):
# frame_cont = cv2.drawContours(frame, contours0, -1, (0,255,0), 3)


# draw the biggest contour (c) in green
#
# debug_img = frame
# debug_img = cv2.resize(debug_img, None, fx=0.3, fy=0.3)





frame = cv2.imread('sample_imgs/p4.jpg')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask_l = np.array([12, 27, 152]) #pink
mask_h = np.array([187, 177, 254])

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
contours0, hierarchy = cv2.findContours(erodtion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
cv2.drawContours(frame, contours0, -1, 255, 3)
c = max(contours0, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("./out.png", frame)