this folder contains some demos of traditional cv methods to be tested.
the main target is to detect hand and output the position of hand, in order to recognition of hand's motion.
member: wang xiaoke, li min

1.25
logic based on Computer Vision
	features: the size of bounding boxes, the center of bouding boxes, ids
	difficulties:
		multi objects: detection cannot identify the box from boxes of last frame
		features not reliable
	to do:
		1-***opencv motion detection/motion trail
			it might be very useful for our project
			key words: motion detection/motion trail/blob/foreground
			the main target is to detect the motion of hand or other things.
			2 methods: subtract the background, or subtract the last frame
		2-***training
			Can we just subtract the background, leave the human and object alone, then train the model?
			if so, how to preprocess the data?
			in this way, we can train a model which is not quite background-sensitive.(but still human sensitive)
		3-***hand detection
			search project on github.
			many projects aim at gusture recognition. 
			also there's a hub using a open-source hands dataset and tf api to train a hand detector. if we plus a basic classifier, will it work?
			
			or
			we can then train a bottle detector. just modify the 1st dataset.
		4-***skin color

1.26
***find useful func in opencv
	opencv official python tutorial
	1-feature matching + homograph
		like sift features and matching. bad perfermance
	2-video analysis--background/foreground segmentation
		has tried some funcs.
		opencv has a module to implement some improved background/foreground segmentation methods: https://docs.opencv.org/3.4.0/d2/d55/group__bgsegm.html
		to do--add more preprocess
		2 open-source code:
			http://blog.csdn.net/stevenke404/article/details/64923855
			http://blog.csdn.net/zhangyonggang886/article/details/51638655
			http://blog.csdn.net/lwplwf/article/details/73551648
			https://www.jianshu.com/p/12533816eddf
			all draw a bounding box, one based on the arc length, one based on the area, the third one consider shadow, 4th one includes shadow and erosion
	3-video analysis--meanshift and camshift
		the demo is okay okay from quran. just use camshift algorithm.
		convert the bgr to hsv. only consider Hue
		opencv/samples/python
			camshift.py
	4-optical flow
		opencv/samples/python/
			opt_flow.py lk_track.py
		how to use optical flow?

discussion with hang and youjian
	from deep learning perspective:
		add another catogory: hand.
		when detection, hand and objects should fit.
		use hand's detection results to predict the motion of hand in/hand out
		--need to annatate new data
	cv:
		may workable
		1-the present frame substracts the last frame
		2-some erosion and dilate preprocess for noise
		3-locate the coord of left blob.
		
		outputs:
		pos of hand(+object), 2 states:hand_in/hand_out
	this work could be done this week.

min:
	env: considering the left_up situation, one man pick up one object
	1-diff 2 continous frames
		gain the motion blob, including object/hand/arm. in complex situations, the passengers may also occur in this mask.
		after modified erosion and dilate, select the left-most connected component as the final blob, output the (xmin, ymin) and rect.
		
		result: 
			when the hands move, the output always responds to the objects, even the object is on the left side;
			however, when the hand is stable, like the moment when man putback the bottle in the cabinet, the the motion cannot be captured/
			and the box will occur on the right side.
		
		func: when the box or coor(xmin,ymin) occurs on the left side for ***many times, we can turn the state to ***hand in.
	2-backg/foreg segmentation
		select the left-most connected component as the final blob.
		
		result:
			in most time, (xmin, ymax) will follow the object. when at the state of hand_in, the coor will definitely inside the left region.
		when using segm method: MOG2, in our situation, human body will be regarded as backg.
		MOG2 backg/foreg segm: ~60fps
		
		gain the blob as same as diff method, but not motion-sensitive. in complex situations, the passengers will occur in this mask for sure.
		valid imfo: when at the state of hand_in and diff method cannot detect a motion blob, the segm can still detect a bbox, of which the coor inside the left region
		all in all, segm can offer a way to decide whether the state is in hand-out

1.27
To do:
	write the basic logic based on Computer Vision
Done:
	write 2 funcs containing diff and foregsegm
	***need to be tested
Diff:
	when run the demo
	***when the motion is slight, no motion blob will be detected. foregsegm also has same problem.
RELEASE:
	logic based on computer vision Version 0.0 is done

1.29
Logic based on Computer Vision Version 1.0:
	problem: the SSD detection is backg-sensitive, which results in many false-positive cases.
	feature: bbox from SSD detection and its id; bbox from segm method
	
	the fist step is to build a test demo to visualize the distance between detection bbox from opencv dnn demo and the segm motion blob.
		***using the (xmin, ymax) coor as the criterion. from testing phase and visualization, we can abtain a simple threshold.
		if the dist bigger than this threshold, we than can conclude those bboxs from SSD detection are false-positive and we can ignore them.
RELEASE:
	cv_logic_v1.py is released.
	Notice: this version is not full-functional. I only concerned about the right region(outside the cabinet)./
	This version can be a intuitive way to understand the difference between bbox of mobilenet-ssd and bbox of fgsemg method.
	I draw a line from (xmin_s, ymax_s) to (xLeftBottom, yRightTop). The distance can be an important feature to decide whether the bbox from mobilenet_ssd is false positive or one we dont care.

the performance analysis: on intel E5 cpu x12
	fgsegm + erosion/dilate operation + diff + erosion/dilate
		fps is not stable. about 55~60 fps
	diff + erosion/dilate operation + ssd detection
		fps is also not stable. about 20 fps
	fgsegm + erosion/dilate operation + diff + erosion/dilate + ssd detection
		16fps
	the speed is not fast enough
	conclusion: 120fps camera make no sense...
one idea: we might can use version 0 as a tracker, and use a classifier to identify the object.
	one problem: the size is variable, when using cnn, should resize the image...I dont think the classification would be good.
	also, this method depend on the motion...if no motion, blow up...

Logic based on Computer Vision Version 2:
	this is the final version.
	assumption: the foreground segmentation and postprocess can capture motion blobs in most cases
		in reality, the fgsegm seems to drop frame in some cases.(due to the modeling of background?)
	basic idea:
		we use subsquential results (or features) to decide which objects are picked up or put back.
		to be more specific:
			when motion PUSH is triggered, we should know what object in the last few frames was detected. (for example, 5 frames. Also the object should/
			be close to motion blob, otherwise it is false positive)
			when motion PULL is triggered, we should know what object in the next few frames was detected.
		constraint of whether a detected object is correct:
			according to logic_cv_v1, the distance between bbox of ssd and bbox of fgsegm should meet the requirement(below the threshold).
		priority:
			the fgsegm motion blob should be placed in the first class
			only after the motion blob is detected, the bbox from mobilenet_ssd can be paired.
			in this case, we can take "nothing in the hand" into account.
			***the states of shopping cart only change when the motion PULL is triggered.

1.30
version 2 is done. may exist few bugs
good idea from wanggang:
	***we can resize/compress the image and do the segm/diff function. let's have a shot!

***cv_logic Version3
wanggang's idea:
	the problem is that the motion PULL and PUSH are triggered by different methods, but not by a unified logic. This may result in the chaos. For example,/
	when the system didn't recognize the PUSH motion, it will trigger 2 PULL motion in sequence.
	also, wanggang tested cv_logic_v0 on another testing video. when upper level of objects are shaked, the fgsegm method will detect this motion blob, even/
	the hand is outside the closet.
	
	***solution:
	like infrared ray detection.
	we only consider the middle region, like(4*480). we take a specific frame as background. when a new frame comes in, we just select same 4*480 region/
	and do the substracion like foreground segmentation.
	after binarization, we can sum up all the light pixels. when the sum is bigger than a threshold, we can conclude that the state is hand_in.

2.1
training log:
	dataset 12_2
		the initial checkpoint is false!!!
	started training again
wrote all the documents needed for training. My job is done.
		
