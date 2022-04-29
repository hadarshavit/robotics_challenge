# import time
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# MIN_MATCH_COUNT = 5
#
# detector = cv2.xfeatures2d.SIFT_create()
# def match(trainKP, trainDesc, queryKP, queryDesc):
#     matches = flann.knnMatch(queryDesc, trainDesc, k=2)
#     goodMatch = []
#     for m, n in matches:
#         if (m.distance < 0.75 * n.distance):
#             goodMatch.append(m)
#     if (len(goodMatch) >= MIN_MATCH_COUNT):
#         tp = []
#         qp = []
#         for m in goodMatch:
#             tp.append(trainKP1[m.trainIdx].pt)
#             qp.append(queryKP[m.queryIdx].pt)
#         tp, qp = np.float32((tp, qp))
#
#         print("Match Found")
#         return tp, qp
#     else:
#         print("Not Enough match found-")
#         print(len(goodMatch), MIN_MATCH_COUNT)
#
# FLANN_INDEX_KDITREE = 0
# flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
# flann = cv2.FlannBasedMatcher(flannParam, {})
#
# trainImg = cv2.imread("sample_imgs/obj.jpg", 0)
# trainKP1, trainDesc1 = detector.detectAndCompute(trainImg, None)
# trainImg1 = cv2.drawKeypoints(trainImg, trainKP1, None, (255, 0, 0), 4)
#
# trainImg = cv2.imread("sample_imgs/obj2.jpg", 0)
# trainKP2, trainDesc2 = detector.detectAndCompute(trainImg, None)
# trainImg2 = cv2.drawKeypoints(trainImg, trainKP2, None, (255, 0, 0), 4)
# # cv
# # plt.show()
# training_data = [(trainKP1, trainDesc1), (trainKP2, trainDesc2)]
#
# cam = cv2.VideoCapture(0)
# for trainKP, trainDesc in training_data:
#     QueryImgBGR = cv2.imread("sample_imgs/p4.jpg")
#     QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
#     queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
#     res = match(trainKP, trainDesc, queryKP, queryDesc)
#     if res:
#         tp, qp = res
#         # H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
#         # h, w = trainImg.shape
#         # trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
#         # # queryBorder = cv2.perspectiveTransform(trainBorder, H)
#         # cv2.polylines(QueryImgBGR, [np.int32(H)], True, (0, 255, 0), 5)
#         img = cv2.drawKeypoints(QueryImgBGR, tp, QueryImgBGR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#         cv2.imshow('result', QueryImgBGR)
#         cv2.imwrite('pp.png', QueryImgBGR)
#     if cv2.waitKey(10) == ord('v'):
#         break
# cam.release()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
# # from matplotlib import pyplot as plt
#
# img1 = cv2.imread('obj.png', 0)          # query Image
# img2 = cv2.imread('p4.jpg',0)  # target Image
#
# # Initiate SIFT detector
# orb = cv2.ORB_create()
#
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1,des2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# good_matches = matches[:10]
#
# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches     ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# matchesMask = mask.ravel().tolist()
# h,w = img1.shape[:2]
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#
# dst = cv2.perspectiveTransform(pts,M)
# dst += (w, 0)  # adding offset
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                singlePointColor = None,
#                matchesMask = matchesMask, # draw only inliers
#                flags = 2)
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)
#
# # Draw bounding box in Red
# img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
#
# cv2.imshow("result", img3)
# cv2.waitKey()

# read images
# import cv2
#
# img1 = cv2.imread('sample_imgs/obj.jpg')
# img2 = cv2.imread('sample_imgs/p4.jpg')
#
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# #sift
# sift = cv2.xfeatures2d.SIFT_create()
#
# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
#
# #feature matching
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#
# matches = bf.match(descriptors_1,descriptors_2)
# matches = sorted(matches, key = lambda x:x.distance)
#
# img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
# cv2.imwrite('out.jpg', img3)










import numpy as np
import cv2

frame = cv2.imread('sample_imgs/p4.jpg')
frame = frame[200:, 50:-50]
# frame = cv2.bitwise_not(frame)
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# kernel = np.ones((5,5), np.uint8)

# cont = cv2.Canny(frame, 230, 260)#110, 130
# n = 1
# img_dilation = cv2.dilate(cont, kernel, iterations=n)
# img_erode = cv2.erode(img_dilation, kernel, iterations=n)
# contours0, hierarchy = cv2.findContours(img_erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
# cv2.drawContours(frame, contours0, -1, 255, 3)
# c = max(contours0, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(c)

# backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=1000)
# fgMask = backSub.apply(frame)
# cv2.imwrite('mask.jpg', fgMask)



# contoursBBS = []
# for contour in contours0:
#     [x, y, w, h] = cv2.boundingRect(contour)
#
#     if w > 100 or h < 8:
#         continue
#
#     contoursBBS.append([x, y, w, h])
# arr = []
# for x,y,w,h in contoursBBS:
#       arr.append((x,y))
#       arr.append((x+w,y+h))
#
# box = cv2.minAreaRect(np.asarray(arr))
# pts = cv2.boxPoints(box) # 4 outer corners
# print(pts)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask_l = np.array([12, 27, 152]) #pink
mask_h = np.array([187, 177, 254])

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



# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imwrite("./out.png", frame)
# x = 242
# y = 454
a = 29 # cm
import numpy as np
alpha = np.arctan(np.abs((480 - y) / (x - 320)))
print(alpha)
alpha += np.pi / 2
print(alpha)

cf = 0.305

x3 = x * cf
y3 = y * cf
x4 = 320 * cf
y4 = 480 * cf

b = np.sqrt(np.power(x4 - x3, 2) + np.power(y4 - y3, 2))

dh = np.sqrt(a**2+b**2-2*a*b*np.cos(alpha))
print(dh)

h = 14.5
do = np.sqrt(dh ** 2+h**2)
print(do)

#x, y, w, h = res
known_width = 7.00
focal_length = 1000

resolution = [440, 380]

print(f'Top left corner cords: {x}, {y}\nDimentions: {w}, {h}')

avg_res = (resolution[0] + resolution[1]) / 2
m = avg_res / focal_length

x = 440 / (resolution[0] / m)
width_pixels_in_cm = w / x

print(width_pixels_in_cm)

distance_mm = (known_width * focal_length) / width_pixels_in_cm  # [mm*mm /mm = mm]

print(distance_mm)

# def findContours(self, image):
#     contour_img = image.copy()
#     vis = image.copy()
#     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
#     _, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
#     contoursBBS = np.array(contoursBBS)
#
#     # reject outliers based on width, height, and area
#     rejectH = self.rejectOutliers(contoursBBS, 3, m=4)
#     rejectW = self.rejectOutliers(rejectH, 2, m=4)
#     rejectA = self.rejectOutliers(rejectW, 4, m=4)
#
#     contourRects = []
#     for c in rejectA:
#         [x, y, w, h, a] = c
#         if w < 9 or h < 15 or a < 300 or a > 6000:
#             continue
#         contourRects.append(c)
#
#     for i, rect in enumerate(contourRects):
#         [x, y, w, h, a] = rect
#         topCenter, bottomCenter = (int((2*x + w)/2), int(y)), (int((2*x + w)/2), int(y+h))
#
#         print("X: {:4d}  Y:  {:4d}  W: {:4d}  H: {:4d}  A: {:4d}".format(x, y, w, h, a))
#
#
#         cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.circle(vis, topCenter, 2, (0, 0, 255), 2)
#         cv2.circle(vis, bottomCenter, 2, (0, 0, 255), 2)