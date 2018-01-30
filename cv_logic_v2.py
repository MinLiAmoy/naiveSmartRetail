'''
logic based totally on Computer Vision.
the basic idea is to detect the position of hand/object using traditional computer vision/
now we combine 2 naive methods: backg/foreg segmentation and two continuous frames' subtraction/
to detect the motion blob of hands/objects.
now we only concern the situation of left_up.
***this version implement a proto of logic based on computer vision. we only consider the bboxs which/
meet the requirement: the dist between bbox of ssd and bbox of fgsegm should smaller than a threshold
notice: I use # need to be modified to imply which params can be changed during the test phase
Author: Min
'''
import cv2 as cv
import numpy as np
import math
import Queue


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

# for opencv dnn module
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
# threshold confidencce of mobile-ssd for detection, need to be modified.
thr_conf = 0.6

# threshold distance of bbox between ssd and fgsegm motion blob
thr_dist = 100   # need to be modified
paired_flag = False


# object Queue to contain the most-recent detected objects which meet the pairing requirement
obj_q = Queue.Queue()
obj_q_maxsize = 5   # need to be modified
# object list, size 13, contain the times that object been detected
obj_list = [0 for i in range(13)]
# the id of the detected object being pushed, follow the classNames
obj_push = 0
# the id of the detected object being pulled, follow the classNames
obj_pull = 0
# flag and count number when motion PULL triggered
cnt_pull = 0
cnt_pull_max = 5
pull_flag = False


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
wrt = cv.VideoWriter('left_up_detection.mp4', cv.VideoWriter_fourcc(*'XVID'), 25, (inWidth, inHeight))



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


def opencv_dnn(frame, net, thr_conf):
    '''
    Target:
        Using opencv_dnn module, input a frame, output the detection result
    Args:
        input:
            present frame;
            net: mobile_ssd;
            threshold of confidence
        output:
            a array: [num_obj * 5], include(idx, xmin, ymin, xmax, ymax)
    '''
    # opencv dnn detection forward
    num_obj = 0
    detections_result = []
    detections = net.forward()
    for i in range(detections.shape[2]):    #ML: detections dim [1, 1, object num, 7]. the last dim, [0]-unknown [1]-class-id [2]-confi [3-6]-scaled coor
        confidence = detections[0, 0, i, 2]
        if confidence > thr_conf:
            num_obj += 1
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * inWidth)
            yLeftBottom = int(detections[0, 0, i, 4] * inHeight)
            xRightTop   = int(detections[0, 0, i, 5] * inWidth)
            yRightTop   = int(detections[0, 0, i, 6] * inHeight)

            if class_id in classNames:
                detections_result += (class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop)

    return detections_result

def compute_dist(coor1, coor2):
    # return the dist from coor1(xmin, ymax) to coor2(xmin, ymax)
    dist = math.sqrt(float((coor1[0]-coor1[0])**2 + (coor1[3]-coor2[3])**2))
    return dist


def segm_dnn_pairing(detections_result, coor_segm, thr_dist):
    '''
    Target:
        pairing the bbox from ssd detection and bbox from fgsegm.
        pairing condition: distance from ssd detected bboxs (xLeftBottom, xRightTop) to segm bbox should below the threshold.
        when multi-objects being detected, we only use the closest ssd bbox.
    Args:
        input:
            detections_result: [5 * number_object], 0-element: id, 1-4(xmin, ymin, xmax, ymax)
            coor_segm: (xmin, ymin, xmax, ymax)
        output:
            pairing flag
            paired object:(id, (coor))
            minimum distance
    '''
    min_dist = 1000
    paired_obj = [None for i in range(5)]
    paired_flag = False
    if len(detections_result) == 0:
        return paired_flag, paired_obj, min_dist
    else:
        for i in range(len(detections_result) / 5):
            coor_ssd = detections_result[5*i+1:5*i+5]
            dist = compute_dist(coor_ssd, coor_segm)
            if dist < min_dist and dist < thr_dist:
                paired_flag = True
                min_dist = dist
                paired_obj = (detections_result[5*i:5*i+5])
        return paired_flag, paired_obj, min_dist


def obj_decision(obj_list):
    '''
    Target:
        find the object detected and paired from the past few frames (5 frames now)
    Args:
        input: a list with size of 13, the element imply the times it been detected and paired
        output: the idx of the decided obj
    To do:
        if multi objs have same count, we just pick the smaller idx, need to be modified
        I am worried about the backg will occupy the list
    '''
    return obj_list.index(max(obj_list))



                

ret, frame = cap.read()
last_frame = None



while (ret):
    if last_frame is None:
        last_frame = frame.copy()
        continue

    # ML: read the size of each frame to 300*300
    net_in = cv.resize(frame, (300, 300))
    blob = cv.dnn.blobFromImage(net_in, inScaleFactor, (300, 300), (meanVal, meanVal, meanVal), swapRB)     #ML: swap RB, then the format becomes rgb
    net.setInput(blob)

    # start timer
    timer = cv.getTickCount()

    # diff method
    # print('using diff method')
    xmin_d, ymin_d, xmax_d, ymax_d = diff(frame, last_frame)
    last_frame = frame.copy()

    # foregsegm method
    # print('using foregsegm')
    xmin_s, ymin_s, xmax_s, ymax_s = fgsegm(frame, fgbg, kernel)

    # opencv dnn detection
    detections_result = opencv_dnn(frame, net, thr_conf)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(frame, "FPS : " + str(int(fps)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)

    # show the detected blob
    # should consider the frame that no detect the motion blob
    if xmin_d < inWidth:
        cv.circle(frame,(int((xmin_d+xmax_d)/2), int((ymin_d+ymax_d)/2)),5, (255,0,0), 5)
        cv.rectangle(frame, (xmin_d, ymin_d), (xmax_d, ymax_d), (255, 0, 0), 3)
    # else:
       # print('no motion blob from diff method')
    if xmin_s < inWidth:
        cv.circle(frame,(int((xmin_s+xmax_s)/2), int((ymin_s+ymax_s)/2)),5, (0,0,255),5)
        cv.rectangle(frame, (xmin_s, ymin_s), (xmax_s, ymax_s), (0, 0, 255), 3)
    # else:
    #     print('no motion blob from segm method')
    
    ###############################################################
    # when segm motion blob is detected, find the paired object and enqueue
    # show the paired bbox
    if xmin_s < inWidth:
        # for PULL motion
        if pull_flag == True:
            cnt_pull += 1

        paired_flag, paired_obj, min_dist = segm_dnn_pairing(detections_result, (xmin_s, ymin_s, xmax_s, ymax_s), thr_dist)
        if obj_q.qsize() == obj_q_maxsize:
            obj_list[obj_q.get()] -= 1

        if paired_flag:
            # only in this conditon, we enqueue.
            obj_q.put(paired_obj[0])
            obj_list[paired_obj[0]] += 1

            (xLeftBottom, yLeftBottom, xRightTop, yRightTop) = paired_obj[1:]

            # draw func
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), 3)
            label = classNames[paired_obj[0]]   # + ": " + str(confidence)
            print('paired, the object is: ' + label)
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv.putText(frame, "dis : " + str(int(min_dist)), (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
            cv.line(frame, (int(xLeftBottom), int(yRightTop)), (int(xmin_s), int(ymax_s)), (0,255,0), 3)
        else:
            obj_q.put(0)
            obj_list[0] += 1

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
    cv.putText(frame, "obj_push : " + str(classNames[obj_push]), (10,120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv.putText(frame, "obj_pull : " + str(classNames[obj_pull]), (10,140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    
    
    # transformation of different state, trigger the motion
    if hand_present != hand_last:
        motion = hand_last - hand_present
        print('hand state changed. the motion is:'+str(motion_dict[motion]))  # -1--push, 1--pull
        hand_last = hand_present

        # the final step, add or delete the object from shopping cart
        if motion == -1:
            obj_push = obj_decision(obj_list)
        else:
            pull_flag = True

    ############################################
    # *** in this case, show what object been pushed and what object been pulled
    if pull_flag and cnt_pull == cnt_pull_max:
        pull_flag = False
        cnt_pull = 0
        obj_pull = obj_decision(obj_list)
        print('PUSH obj:' + str(obj_push))
        cv.putText(frame, "PUSH obj : " + str(classNames[obj_push]), (10,160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
        print('PULL obj:' + str(obj_pull))
        cv.putText(frame, "PULL obj : " + str(classNames[obj_pull]), (10,180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)



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

    
