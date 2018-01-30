'''
logic based totally on Computer Vision--Version 0.0
the basic idea is to detect the position of hand/object using traditional computer vision/
now we combine 2 naive methods: backg/foreg segmentation and two continuous frames' subtraction/
to detect the motion blob of hands/objects.
now we only concern the situation of left_up.
notice: I use # need to be modified to imply which params can be changed during the test phase
Author: Min
'''
import cv2 as cv
import numpy as np


# 2 important definition and output: states and motion
# hand_s: 0-hands_out when hand in the right region, 1-hands_in when hand in the left region
# motion: default value 0--no motion, -1--when hand_s from 0->1--push, 1--hand_s from 1->0 pull
hand_last = 0
hand_present = 0
motion = 0
motion_dict = {-1:'PUSH', 0:'None', 1:'PULL'}

# count number of diff and segm
cnt_d = 0   # count num for diff
cnt_s = 0   # count num for segm

# the size of original image
inWidth = 640
inHeight = 480

# 2 coor to dicide the left region and the right region. draw a line from (xmin, 0) to (xmax, 480)
# need to be modified
xmin_region = 340.0
ymin_region = 0.0
xmax_region = 270.0
ymax_region = 480.0
# the func of line: y = kx + b, need to be modified, the if flow did not work
k = (ymax_region - ymin_region)/(xmax_region - xmin_region) # k = -4.8
# print(k)
b = ymax_region - (ymax_region - ymin_region)/(xmax_region - xmin_region)*xmax_region # b = 1776
# print(b)

# whether to write the video for analysis
Write_Video = True

cap = cv.VideoCapture("left_up.mp4")
wrt = cv.VideoWriter('left_up_motion.mp4', cv.VideoWriter_fourcc(*'XVID'), 25, (inWidth, inHeight))



def diff(present_frame, last_frame):
    '''
    Target: 
        predict the position of moving object by diff the 2 continious frames and some cv operation
        we only care about the left-most motion blob, which indicate how far the hand/object reach
        ***when the box or coor(xmin,ymin) occurs on the left side for ***many times/ 
        we can turn the state to ***hand in.
    Agrs: 
        inputs:
            present frame
            last_frame
        outputs:
            bbox:(xmin,ymin,xmax,ymax) of most-left motion blob
    To do: copy the present_gray to last_gray, in oder to get rid of color conversion
    '''
    # var to save the xmin coor
    xmin = inWidth
    ymin = None
    xmax = None
    ymax = None

    present_gray = cv.cvtColor(present_frame, cv.COLOR_BGR2GRAY)
    present_gray = cv.GaussianBlur(present_gray, (21, 21), 0)
    last_gray = cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)
    last_gray = cv.GaussianBlur(last_gray, (21, 21), 0)

    frame_delta = cv.absdiff(last_gray, present_gray)
    frame_delta = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1] # the threshold should be modified
    # img show the diff img without operation
    # cv.imshow('diff_1', frame_delta)
    
    #operation, need to be modified
    # frame_delta = cv.erode(frame_delta, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # frame_delta = cv.dilate(frame_delta, cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 3)), iterations=2)
    frame_delta = cv.erode(frame_delta,None, iterations=2)
    frame_delta = cv.erode(frame_delta, None, iterations=2)
    frame_delta = cv.erode(frame_delta, None, iterations=2)
    frame_delta = cv.erode(frame_delta, None, iterations=1)
    frame_delta = cv.dilate(frame_delta, None, iterations=10)
    # img show the diff img with operation
    # cv.imshow("diff_2", frame_delta)

    # find be connected components and draw the rect bbox
    (_, cnts, _) = cv.findContours(frame_delta.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
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
        # cv.rectangle(frame_delta, (x, y), (x + w, y + h), (0, 255, 0), 5) # when comment these drawing func, gain a higher speed
    # img show the diff img with operation and rect bbox
    # cv.imshow("diff_3", frame_delta)
    return xmin, ymin, xmax, ymax


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


# select the subtractor
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()  # MOG
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()    # GMG-use fisrt 120 frames to model the background by default
fgbg = cv.createBackgroundSubtractorMOG2()    # MOG2
# fgbg = cv.createBackgroundSubtractorKNN() # KNN
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

# the vars to contain the bbox output of diff method
xmin_d = None
ymin_d = None
xmax_d = None
ymax_d = None
# the vars to contain the bbox output of segm method
xmin_s = None
ymin_s = None
xmax_s = None
ymax_s = None



ret, frame = cap.read()
last_frame = None



while (ret):
    if last_frame is None:
        last_frame = frame.copy()
        continue

    # cv.imshow('frame', frame)

    # start timer
    timer = cv.getTickCount()


    # diff method
    # print('using diff method')
    xmin_d, ymin_d, xmax_d, ymax_d = diff(frame, last_frame)
    last_frame = frame.copy()

    # foregsegm method
    # print('using foregsegm')
    xmin_s, ymin_s, xmax_s, ymax_s = fgsegm(frame, fgbg, kernel)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(frame, "FPS : " + str(int(fps)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)

    # show the detected blob
    # should consider the frame that no detect the motion blob
    if xmin_d < inWidth:
        cv.circle(frame,(int((xmin_d+xmax_d)/2), int((ymin_d+ymax_d)/2)),5, (255,0,0), 5)
        cv.rectangle(frame, (xmin_d, ymin_d), (xmax_d, ymax_d), (255, 0, 0), 3)
    else:
        print('no motion blob from diff method')
    if xmin_s < inWidth:
        cv.circle(frame,(int((xmin_s+xmax_s)/2), int((ymin_s+ymax_s)/2)),5, (0,0,255),5)
        cv.rectangle(frame, (xmin_s, ymin_s), (xmax_s, ymax_s), (0, 0, 255), 3)
    else:
        print('no motion blob from segm method')
    # img show with detected motion blob
    # cv.imshow('frame', frame)

    #################################################################
    # ***logic func
    
    # every frame has a loss:-1
    cnt_s -= 1
    cnt_d -= 1

    # considering the diff method to decide whether the state is 1--hands_in
    # To do: add info from diff to cross-verify
    if xmin_d < inWidth:
        if (4.8*xmin_d+ymin_d-1776) < 0:    # need to be modified, use k and b didn't work
            cnt_d += 3  # when detected in the left region, reward 3, need to be modified
    else:
        cnt_d -= 1  # else loss -1
    if cnt_d >= 5:
        hand_present = 1
        cnt_d = 5   # need to be modified

    # considering the segm method to decide whether the state is 0--hands_out
    if xmin_s < inWidth:
        if (4.8*xmin_s+ymin_s-1776) > 0:    # need to be modified, use k and b didn't work
            cnt_s += 2  # when detected in the right region, reward 2, need to be modified
    else:
        cnt_s -= 1  # else loss -1
    if cnt_s >= 5:
        hand_present = 0
        cnt_s = 5   # need to be modified

    if cnt_s < 0:
        cnt_s = 0
    if cnt_d < 0:
        cnt_d = 0

    # visualize the count number and state
    cv.putText(frame, "cnt_d : " + str(int(cnt_d)), (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv.putText(frame, "cnt_s : " + str(int(cnt_s)), (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv.putText(frame, "state : " + str(int(hand_present)), (10,80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    # transformation of different state
    if hand_present != hand_last:
        motion = hand_last - hand_present
        print('hand state changed. the motion is:')  # -1--push, 1--pull
        print(motion_dict[motion])
        hand_last = hand_present
    # draw the line to divide the img into the left region and right region
    cv.line(frame, (int(xmin_region), int(ymin_region)), (int(xmax_region), int(ymax_region)), (0,255,0))
    # img show with detected motion blob
    cv.imshow('frame', frame)

    if Write_Video:
        wrt.write(frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    ret, frame = cap.read()
    
cap.release()
cv.destroyAllWindows()

    
