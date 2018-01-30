''' 
from opencv official python tutorial
meanshift and camshift demo
'''
import numpy as np
import cv2 as cv


cap =cv.VideoCapture('../left_up.mp4')

ret, frame = cap.read()

imgCrop =cv.selectROI(frame)
r,h,c,w =int(imgCrop[1]),int(imgCrop[3]),int(imgCrop[0]),int(imgCrop[2])
track_window =(c,r,w,h)
roi =frame[r:r+h,c:c+w]
# cv.imshow("roi",roi)
# cv.waitKey()

hsv_roi =cv.cvtColor(roi,cv.COLOR_BGR2HSV)	# convert roi from bgr to hsv
mask =cv.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist =cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

term_crit =(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,1)

while True:
    ret,frame =cap.read()
    if ret ==True:
        hsv =cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst =cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	'''
	# from quran, a poly not a rect
        ret,track_window =cv.CamShift(dst,track_window,term_crit)
        pts =cv.boxPoints(ret)
        pts =np.int0(pts)
        img2 =cv.polylines(frame,[pts],True,255,2)'''
	# apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow("img2",img2)
        k =cv.waitKey(30) &0xff
        if k==27:
            break
    else:
        break
