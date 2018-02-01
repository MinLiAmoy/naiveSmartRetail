'''
logic based totally on Computer Vision--Version 3.0
the basic idea is to detect whether hand/object is crossed the line(a narrow region), like infrared ray detection.
when cross the line, we conclude that the state is hand_in, otherwise hand_out
now we only concern the situation of left_up.
***wanggang has implemented this func in the testing code, pls refer to his codes
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

# the size of original image
inWidth = 640
inHeight = 480

# define the line region
x_line_min = 300
y_line_min = 0
x_line_max = 320
y_line_max = 480

# threshold of area
threshold_area = 20

# whether to write the video for analysis
Write_Video = False

cap = cv.VideoCapture("left_up.mp4")
wrt = cv.VideoWriter('left_up_line.mp4', cv.VideoWriter_fourcc(*'XVID'), 25, (inWidth, inHeight))



def diff(present_frame, template):
    '''
    Target: 
        predict whether the hand is inside the closet. the decision policy is to detect whether hand/object
        is crossed the line. we use template substraction and turn the delta into binary. when the sum of light/
        pixels is up to threshold, then we can say that the state is on hands_in
    Agrs: 
        inputs:
            present frame
            template
        outputs:
            hand'state: 0-hands_out, 1-hands_in
    To do: copy the present_gray to last_gray, in oder to get rid of color conversion
    '''
    hand_state = 0

    present_gray = cv.cvtColor(present_frame, cv.COLOR_BGR2GRAY)
    present_gray = cv.GaussianBlur(present_gray, (21, 21), 0)
    last_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    last_gray = cv.GaussianBlur(last_gray, (21, 21), 0)

    frame_delta = cv.absdiff(last_gray, present_gray)
    frame_delta = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1] # the threshold should be modified
    # img show the diff img without operation
    cv.imshow('diff_1', frame_delta)
    
    #operation, need to be modified
    # frame_delta = cv.erode(frame_delta, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # frame_delta = cv.dilate(frame_delta, cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # frame_delta = cv.erode(frame_delta,None, iterations=2)
    # frame_delta = cv.erode(frame_delta, None, iterations=2)
    # frame_delta = cv.erode(frame_delta, None, iterations=2)
    # frame_delta = cv.erode(frame_delta, None, iterations=1)
    # frame_delta = cv.dilate(frame_delta, None, iterations=10)
    # img show the diff img with operation
    # cv.imshow("diff_2", frame_delta)

    # find be connected components and draw the rect bbox
    (_, cnts, _) = cv.findContours(frame_delta.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv.contourArea(c) < threshold_area: # need to be tested
            continue
        else:
            hand_state = 1
            (x, y, w, h) = cv.boundingRect(c)
            #print(cv.contourArea(c))
            cv.rectangle(frame_delta, (x, y), (x + w, y + h), (0, 255, 0), 5) # when comment these drawing func, gain a higher speed
            # img show the diff img with operation and rect bbox
            cv.imshow("diff_3", frame_delta)
    return hand_state

ret, frame = cap.read()
template = None


while (ret):

    if template is None:
        template = frame.copy()
        template = template[y_line_min:y_line_max, x_line_min:x_line_max]
        cv.imshow('template', template)
        continue

    # cv.imshow('frame', frame)

    # start timer
    timer = cv.getTickCount()

    # diff method
    # print('using diff method')
    hand_present = diff(frame[y_line_min:y_line_max, x_line_min:x_line_max], template)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(frame, "FPS : " + str(int(fps)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)

    #################################################################
    # ***logic func
    

    # visualize the count number and state
    cv.putText(frame, "state : " + str(int(hand_present)), (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    # transformation of different state
    if hand_present != hand_last:
        motion = hand_last - hand_present
        print('hand state changed. the motion is:')  # -1--push, 1--pull
        print(motion_dict[motion])
        hand_last = hand_present

    # draw the line to divide the img into the left region and right region
    cv.rectangle(frame, (x_line_min, y_line_min), (x_line_max, y_line_max), (0,255,0))
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

    
