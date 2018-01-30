#!usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import time

camera = cv2.VideoCapture("test_video/left_up.avi")
width = int(camera.get(3))
height = int(camera.get(4))

firstFrame = None



def detect_video(video):
    camera = cv2.VideoCapture(video)
    history = 5    # 训练帧数

    bs = cv2.createBackgroundSubtractorMOG2()()  # 背景减除器，设置阴影检测
    frames = 0

    while True:
        res, frame = camera.read()

        if not res:
            break

        fg_mask = bs.apply(frame)   # 获取 foreground mask

        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 25, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 获取矩形框边界坐标
            #x, y, w, h = cv2.boundingRect(dilated)
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            #area = cv2.contourArea(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)
        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()




left_x = 640
left_y = 0
left_w = 0
left_h = 0

bs = cv2.createBackgroundSubtractorMOG2()  # 背景减除器，设置阴影检
while True:
    res, frame = camera.read()
    time1 = time.time()

    if not res:
        break
    fg_mask = bs.apply(frame)  # 获取 foreground mask


    thresh = cv2.threshold(fg_mask.copy(), 25, 255, cv2.THRESH_BINARY)[1]
    # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    #dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=10)
    # 获取所有检测框
    image, contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 获取矩形框边界坐标
        #x, y, w, h = cv2.boundingRect(thresh)
        x, y, w, h = cv2.boundingRect(c)
        if x < left_x:
            left_x = x
            left_y = y
            left_w = w
            left_h = h

    #print(time.time() - time1)
    #cv2.circle(frame, (left_x + int(left_w / 2), left_y + int(left_h / 2)), 9, (255, 0, 0), 5)
    if left_x > 300:
        print("out")
    else:
        print("in")
    cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 5)
    left_x = 640
    cv2.imshow("detection", frame)
    cv2.imshow("back", fg_mask)
    cv2.imshow("thresh", thresh)
    if cv2.waitKey(30) & 0xff == 27:
        break
camera.release()







# bg = cv2.createBackgroundSubtractorMOG2()
# while True:
#     (grabbed, frame) = camera.read()
#     fg_mask = bg.apply(frame)
#     # fg_mask = bs.apply(frame)  # 获取 foreground mask
#     # if frames < history:
#     #     frames += 1
#     #     continue
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)
#     #gray = cv2.GaussianBlur(fg_mask, (21, 21), 0)
#     if firstFrame is None:
#         firstFrame = gray
#         continue
#
#     frameDelta = cv2.absdiff(firstFrame, gray)
#     thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
#     #thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_OTSU)[1]
#     print("thresh")
#     print(thresh.shape)
#     print(thresh.shape[0])
#     print(thresh.shape[1])
#     # for i in range(thresh.shape[0]):
#     #     for j in range(thresh.shape[1])
#     cv2.imshow("thresh1", thresh)
#     # th = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
#     # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
#     thresh = cv2.erode(thresh,None, iterations=2)
#     thresh = cv2.erode(thresh, None, iterations=2)
#     thresh = cv2.erode(thresh, None, iterations=2)
#     thresh = cv2.erode(thresh, None, iterations=1)
#     thresh = cv2.dilate(thresh, None, iterations=10)
#     cv2.imshow("thresh3", thresh)
#     # for i in range(thresh.shape[0]):
#     #     for j in range(thresh.shape[1]):
#     #         pixel = thresh[i, j]
#     #         print("pixel")
#     #         print(pixel)
#
#     #(x, y, w, h) = cv2.boundingRect(thresh.copy())
#
#     # print(cv2.contourArea(c))
#     #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
#
#     (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts:
#         # if cv2.contourArea(c) < 0:
#         #     continue
#         (x, y, w, h) = cv2.boundingRect(c)
#
#         if x < left_x:
#             left_x = x
#             left_y = y
#             left_w = w
#             left_h = h
#
#         #print(cv2.contourArea(c))
#
#     cv2.circle(frame,(left_x + int(left_w/2), left_y + int(left_h/2)),9, (255,0,0), 5)
#     cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 5)
#     left_x = 640
#
#     cv2.imshow("frame", frame)
#     cv2.waitKey(100)
#     firstFrame = gray.copy()
# camera.release()
# cv2.destroyAllWindows()


# if __name__ == '__main__':
#     video = 'test_video/left_up.avi'
#     detect_video(video)