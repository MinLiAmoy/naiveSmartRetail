# This script is used to demonstrate MobileNet-SSD network using OpenCV deep learning module.
#
# It works with model taken from https://github.com/chuanqi305/MobileNet-SSD/ that
# was trained in Caffe-SSD framework, https://github.com/weiliu89/caffe/tree/ssd.
# Model detects objects from 20 classes.
#
# Also TensorFlow model from TensorFlow object detection model zoo may be used to
# detect objects from 90 classes:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# Text graph definition must be taken from opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
import numpy as np
import argparse

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

usb_cameras=["/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0",
"/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0",
"/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0",
"/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to run MobileNet-SSD object detection network '
                    'trained either in Caffe or TensorFlow frameworks.')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                      help='Path to text network file: '
                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                           'ssd_mobilenet_v1_coco.pbtxt from opencv_extra for TensorFlow model')
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                     help='Path to weights: '
                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                          'frozen_inference_graph.pb from TensorFlow.')
    parser.add_argument("--model_type", default=0, type=int,
                        help="caffe 0 or tensorflow 1")
    parser.add_argument("--thr", default=0.6, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    if args.model_type == 0:
        net = cv.dnn.readNetFromCaffe(args.prototxt, args.weights)
        swapRB = False
        classNames = classNames = {0: 'background',
                  1: '010001', 2: '002004', 3: '006001', 4: '007001', 5: '008001', 6: '009001',
                  7: '001001', 8: '002001', 9: '002002', 10: '003001', 11: '003002',
                  12: '003003'}
    else:
        assert(args.model_type == 1)
        net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
        swapRB = True
        classNames = {0: 'background',
                  1: '010001', 2: '002004', 3: '006001', 4: '007001', 5: '008001', 6: '009001',
                  7: '001001', 8: '002001', 9: '002002', 10: '003001', 11: '003002',
                  12: '003003'}

    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(usb_cameras[0])
        #0:左上 1:左下 2:右上 3:右下

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()     #Youjian: in bgr format

        # ML: read the size of each frame
        cols = frame.shape[1]
        rows = frame.shape[0]
        tensor = cv.resize(frame, (inWidth, inHeight))
        blob = cv.dnn.blobFromImage(tensor, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)     #ML: swap RB, then the format becomes rgb
        net.setInput(blob)
        
        # start timer
        timer = cv.getTickCount()

        detections = net.forward()
	    # frame = cv.resize(frame, (cols, rows))

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        cv.putText(frame, "FPS : " + str(int(fps)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2);
        '''if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]
        # ML: the original version is to crop the center of image.
        cols = frame.shape[1]
        rows = frame.shape[0]'''
        

        for i in range(detections.shape[2]):    #ML: detections dim [1, 1, object num, 7]. the last dim, [0]-unknown [1]-class-id [2]-confi [3-6]-scaled coor
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])

		

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
	    # frame = cv.resize(frame, (cols, rows))
        cv.imshow("detections", frame)
        if cv.waitKey(1) >= 0:
            break
