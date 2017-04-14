import cv2
import time
import numpy as np
import pandas as pd

IMG_PATH_TEMPLATE_1 = 'coke.png'		#Path to reference image for object 1
IMG_PATH_TEMPLATE_2 = 'bisleri.png'		#Path to reference image for object 2
THRESHOLD_TEMPLATE_1 = 0.6			#Matching threshold for image 1 ~Range 0-1
THRESHOLD_TEMPLATE_2 = 0.6			#Matching threshold for image 1 ~Range 0-1
IMG_PATH_INPUT = 'input.png'			#Path to input image

img_rgb = cv2.imread(IMG_PATH_INPUT)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
capture = cv2.VideoCapture('Inventory Tracker')
capture
cap = cv2.VideoCapture(0)			#Getting the video object of camera interface, 0 for default camera 
		
def extract_matching_points(template_img_path, threshold):		#Method to extract the matching points from the input image
	template = cv2.imread(template_img_path,0)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	loc = np.where( res >= threshold )
	collectionlist = list()
	for pt in zip(*loc[::-1]):
		boxes=(pt[0],pt[1],pt[0] + w, pt[1] + h)
		collectionlist.append(list(boxes))
        return np.array(collectionlist)

def matching_points_suppression(boxes):					#Method to supress the duplicate matching points
	template_match_count = 0
       	if len(boxes) == 0:
		return 0
        if boxes.dtype.kind == "i":
	        boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
	y2 = boxes[:,3]
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
	while len(idxs) > 0:
                last = len(idxs) - 1
        	i = idxs[last]
                pick.append(i)
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
	        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        	xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
	        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > 0.3)[0])))
        for pt in boxes[pick].astype("int"):
		cv2.rectangle(img_rgb, (pt[0],pt[1]), (pt[2], pt[3]), (0,0,255), 2)
		template_match_count += 1
	return template_match_count

count_list = [0,0]			#List to store the object detection count
start_time = time.time()
while True:
	template1_match_count = 0
	template2_match_count = 0
	ret,im = cap.read()
	waited = time.time() - start_time
	if waited >= 5:
		#cv2.imwrite('input.png', im)
		#boxes = extract_matching_points(IMG_PATH_TEMPLATE_1,THRESHOLD_TEMPLATE_1)
		#count_list.insert(0, matching_points_suppression(boxes))
		boxes = extract_matching_points(IMG_PATH_TEMPLATE_2,THRESHOLD_TEMPLATE_2)
		count_list.insert(1, matching_points_suppression(boxes))
		start_time = time.time()
	if cv2.waitKey(10) == 27:
    		break
        cv2.putText(im, "Object1 count  :  "+`count_list[0]`, (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
	cv2.putText(im, "Object2 count   :  "+`count_list[1]`, (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)	
	cv2.imshow('Inventory Tracker',im)
