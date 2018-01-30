''' 
test all the subtractor provided by opencv3.4
from opencv official python tutorial
'''

import numpy as np
import cv2 as cv
cap = cv.VideoCapture('../left_up.mp4')

# select the subtractor

# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()	# MOG

fgbg = cv.bgsegm.createBackgroundSubtractorGMG()	# GMG-use fisrt 120 frames to model the background by default

# fgbg = cv.createBackgroundSubtractorMOG2()	# MOG2

# fgbg = cv.createBackgroundSubtractorKNN()	# KNN

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))


ret, frame = cap.read()
while(ret):
    
    fgmask = fgbg.apply(frame)
    # open operation to smooth the noise
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    ret, frame = cap.read()
cap.release()
cv.destroyAllWindows()
