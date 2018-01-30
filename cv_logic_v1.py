'''
logic based totally on Computer Vision--Version 1.0
the basic idea is to detect the position of hand/object using traditional computer vision/
now we combine 2 naive methods: backg/foreg segmentation and two continuous frames' subtraction/
to detect the motion blob of hands/objects.
now we only concern the situation of left_up.
***this version only implement the foregsemg func, caculate dis between foregsemg bbx and ssd bbx
reference: opencv_dnn_demo/mobilenet_ssd_python.py
notice: I use # need to be modified to imply which params can be changed during the test phase
Author: Min
'''
import cv2 as cv
import numpy as np
import math


inWidth = 640
inHeight = 480
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
# threshold confidencce of mobile-ssd for detection
thr = 0.6

# whether to write the video for analysis
Write_Video = True
cap = cv.VideoCapture('left_up.mp4')
wrt = cv.VideoWriter('left_up_dist.mp4', cv.VideoWriter_fourcc(*'XVID'), 25, (inWidth, inHeight))



def fgsegm(frame, fgbg, kernel):
    '''
    Target:
        preditct the most left motion blob as same as diff, but not motion-sensitive
        when a man do the motion of push or select the object in the cabinet, the hand/object is stable/
        in this way, diff become invalid and messed up. but backgsegm can still detect this blob.
        we are considering using backgsegm to turn the state to hands_out
    Args:
        inputs:
            present frame,
            fgbg: the subtractor, should be define at the main func
            kernel: from many tutorial, the kernel is applied to remove noise
        outputs:
            bbox(xmin, ymin, xmax, ymax)
    To do: try several morphological filtering operations like closing and opening/
        in order to get an optimal performance
    '''
    # var to save the xmin coor
    xmin = inWidth
    ymin = None
    xmax = None
    ymax = None

    fgmask = fgbg.apply(frame)
    # open operation to smooth the noise
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    # img show the fgsegm with default operation
    # cv.imshow('fgsegm_1',fgmask)

    fgmask = cv.threshold(fgmask, 244, 255, cv.THRESH_BINARY)[1] # wipe the shadow
    # img show the diff img withou operation, containing shadow
    # cv.imshow('fgmask_2', fgmask)
    
    #operation, need to be modified
    # fgmask = cv.erode(fgmask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # fgmask = cv.dilate(fgmask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 3)), iterations=2)
    fgmask = cv.erode(fgmask,None, iterations=2)
    fgmask = cv.erode(fgmask, None, iterations=2)
    fgmask = cv.erode(fgmask, None, iterations=2)
    fgmask = cv.erode(fgmask, None, iterations=1)
    fgmask = cv.dilate(fgmask, None, iterations=10)
    # img show the diff img with operation
    # cv.imshow("diff_3", fgmask)

    # find be connected components and draw the rect bbox
    (_, cnts, _) = cv.findContours(fgmask.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv.contourArea(c) < 20: # need to be tested
            continue
        (x, y, w, h) = cv.boundingRect(c)
        #print(cv.contourArea(c))
        if x < xmin:
            xmin = x
            ymin = y
            xmax = x+w
            ymax = y+h
        # cv.rectangle(fgmask, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # img show the diff img with operation and rect bbox
    # cv.imshow("diff_4", fgmask)
    return xmin, ymin, xmax, ymax

# *** for fgsemg part
# select the subtractor
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()  # MOG
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()    # GMG-use fisrt 120 frames to model the background by default
fgbg = cv.createBackgroundSubtractorMOG2()    # MOG2
# fgbg = cv.createBackgroundSubtractorKNN() # KNN
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

# vars to contain the bbox output of segm method
xmin_s = None
ymin_s = None
xmax_s = None
ymax_s = None
# read the model trained on tensorflow object detection api, two files should be included:*.pb, *.pbtxt
net = cv.dnn.readNetFromTensorflow('opencv_dnn_demo/frozen_inference_graph.pb', 'opencv_dnn_demo/ssd_mobilenet_v1_coco.pbtxt')
swapRB = True
# cvlassNames = {0: 'background',
#               1: '010001', 2: '002004', 3: '006001', 4: '007001', 5: '008001', 6: '009001',
#               7: '001001', 8: '002001', 9: '002002', 10: '003001', 11: '003002',
#               12: '003003'}
# class dict
classNames = {0: 'background',
              1: 'vita', 2: 'orange', 3: 'redbull', 4: 'move', 5: 'blacktea', 6: 'pear',
              7: 'water', 8: 'sprite', 9: 'cola', 10: 'milktea', 11: 'ming',
              12: 'needle'}


# Capture frame-by-frame
ret, frame = cap.read()     #Youjian: in bgr format

while ret:
    # ML: read the size of each frame to 300*300
    net_in = cv.resize(frame, (300, 300))
    blob = cv.dnn.blobFromImage(net_in, inScaleFactor, (300, 300), (meanVal, meanVal, meanVal), swapRB)     #ML: swap RB, then the format becomes rgb
    net.setInput(blob)
    
    # start timer
    timer = cv.getTickCount()
    detections = net.forward()

    # fgsegm motion detection part
    xmin_s, ymin_s, xmax_s, ymax_s = fgsegm(frame, fgbg, kernel)

    if xmin_s < inWidth:
        cv.circle(frame,(int((xmin_s+xmax_s)/2), int((ymin_s+ymax_s)/2)),5, (255,0,0), 5)
        cv.rectangle(frame, (xmin_s, ymin_s), (xmax_s, ymax_s), (255, 0, 0), 3)
    else:
        print('no motion blob from segm method')

    # timer quit earlier, neeed to be modified
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(frame, "FPS : " + str(int(fps)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    

    for i in range(detections.shape[2]):    #ML: detections dim [1, 1, object num, 7]. the last dim, [0]-unknown [1]-class-id [2]-confi [3-6]-scaled coor
        confidence = detections[0, 0, i, 2]
        if confidence > thr:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * inWidth)
            yLeftBottom = int(detections[0, 0, i, 4] * inHeight)
            xRightTop   = int(detections[0, 0, i, 5] * inWidth)
            yRightTop   = int(detections[0, 0, i, 6] * inHeight)

            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), 3)
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                # draw the line from (xLeftBottom, yRightTop) to (xmin_s, ymax_s)
                if xmin_s < inWidth:
                	cv.line(frame, (int(xLeftBottom), int(yRightTop)), (int(xmin_s), int(ymax_s)), (0,255,0), 3)
                	# To do: use numpy.linalg
                	dist = math.sqrt(float((xLeftBottom-xmin_s)**2 + (yRightTop-ymax_s)**2))
                    # print(str(dist))
                	cv.putText(frame, "dis: " + str(dist), (10,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv.imshow("frame", frame)
    if Write_Video:
        wrt.write(frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    ret, frame = cap.read()
    
cap.release()
cv.destroyAllWindows()
