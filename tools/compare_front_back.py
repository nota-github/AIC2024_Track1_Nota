import cv2
import sys
sys.path.append('.')
from reids.fastreid.models.model import ReID
from trackers.multicam_tracker.matching import embedding_distance

front = cv2.imread('tools/13_0.jpg')
h,w,_ = front.shape
bbox_front = [0,0,w,h]
print(bbox_front)
back = cv2.imread('tools/13_1.jpg')
h,w,_ = back.shape
bbox_back = [0,0,w,h]
print(bbox_back)

ReID.initialize(max_batch_size=1)
reid = ReID()

front_feat = reid.run(front, [bbox_front], 1)
back_feat = reid.run(back, [bbox_back], 1)

dist = embedding_distance(front_feat, back_feat) / 2.0
print('embedding dist: ', dist)

ReID.finalize()