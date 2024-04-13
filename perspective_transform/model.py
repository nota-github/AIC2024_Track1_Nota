import cv2 
import numpy as np
import json
from scipy.stats import norm


KEYPOINTS = [
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "head",
                "neck"
            ]


class PerspectiveTransform:
    def __init__(self, calibration):
        with open(calibration, 'r') as f:
            matrixes = json.load(f)
        self.camera_projection_matrix_inv = np.array(matrixes['camera projection matrix']).astype(np.float32)
        self.homography_matrix_inv = np.array(matrixes['homography matrix']).astype(np.float32)
        self.ratio = np.array([3.5, 7., 2., 1.6, 7., 5., 3.])
        self.pose_thr = 0.3
        
        self.initialize()
    
    def initialize(self):
        self.homography_matrix = np.linalg.inv(self.homography_matrix_inv)
    
    def run(self, tracker, new_ratio, cam_id=None):
        for i in range(len(tracker.tracked_stracks)):
            tlbr = tracker.tracked_stracks[i].tlbr.tolist()
            pose = tracker.tracked_stracks[i].pose
            if new_ratio is not None:
                self.ratio = self.ratio * 0.9 + new_ratio * 0.1
            # print('ratio: ', self.ratio)
            
            if pose is None:
                w = abs(tlbr[0] - tlbr[2])
                bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + w * self.ratio[0], 1]
                bottom_left = [tlbr[0], tlbr[1] + w * self.ratio[0], 1]
                
            else:
                keys = dict(zip(KEYPOINTS, pose['keypoints'].tolist()))
                m, M, min_key, max_key = 1000, 1000, '', ''
                for key in keys.keys():
                    if keys[key][2] > self.pose_thr:
                        if min(abs(keys[key][1] - tlbr[3]), m) == abs(keys[key][1] - tlbr[3]):
                            m = abs(keys[key][1] - tlbr[3])
                            min_key = key
                        if min(abs(keys[key][1] - tlbr[1]), M) == abs(keys[key][1]-tlbr[1]):
                            M = abs(keys[key][1] - tlbr[1])
                            max_key = key
                                    
                h = tlbr[3] - tlbr[1]
                if min_key == '':
                    w = abs(tlbr[0] - tlbr[2])
                    bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + w * self.ratio[0], 1]
                    bottom_left = [tlbr[0], tlbr[1] + w * self.ratio[0], 1]
                    
                elif 'ankle' in min_key:
                    bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3], 1]
                    bottom_left = [tlbr[0], tlbr[3], 1]
                    
                elif 'head' in min_key: 
                    bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * self.ratio[1], 1]
                    bottom_left = [tlbr[0], tlbr[3] + h * self.ratio[1], 1]
                
                elif (keys[min_key][1] + h/4) < tlbr[3]:
                    if ('head' in max_key) or ('neck' in max_key):
                        key_gap = abs(keys['head'][1] - keys[min_key][1])
                        if ('hip' in min_key) or ('wrist' in min_key):
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + key_gap * self.ratio[2], 1]
                            bottom_left = [tlbr[0], tlbr[1] + key_gap * self.ratio[2], 1]
                        elif 'knee' in min_key:
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + key_gap * self.ratio[3], 1]
                            bottom_left = [tlbr[0], tlbr[1] + key_gap * self.ratio[3], 1]
                        elif 'neck' in min_key: 
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + key_gap * self.ratio[4], 1]
                            bottom_left = [tlbr[0], tlbr[1] + key_gap * self.ratio[4], 1]
                        elif 'shoulder' in min_key:
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + key_gap * self.ratio[5], 1]
                            bottom_left = [tlbr[0], tlbr[1] + key_gap * self.ratio[5], 1]
                        elif 'elbow' in min_key:
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[1] + key_gap * self.ratio[6], 1]
                            bottom_left = [tlbr[0], tlbr[1] + key_gap * self.ratio[6], 1]
                        else:
                            bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3], 1]
                            bottom_left = [tlbr[0], tlbr[3], 1]
                    else:
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3], 1]
                        bottom_left = [tlbr[0], tlbr[3], 1]
                        
                else:
                    if ('hip' in min_key) or ('wrist' in min_key):
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * (self.ratio[2] - 1.), 1]
                        bottom_left = [tlbr[0], tlbr[3] + h * (self.ratio[2] - 1.), 1]
                    elif 'knee' in min_key:
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * (self.ratio[3] - 1.), 1]
                        bottom_left = [tlbr[0], tlbr[3] + h * (self.ratio[3] - 1.), 1]
                    elif 'neck' in min_key: 
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * (self.ratio[4] - 1.), 1]
                        bottom_left = [tlbr[0], tlbr[3] + h * (self.ratio[4] - 1.), 1]
                    elif 'shoulder' in min_key:
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * (self.ratio[5] - 1.), 1]
                        bottom_left = [tlbr[0], tlbr[3] + h * (self.ratio[5] - 1.), 1]
                    elif 'elbow' in min_key:
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3] + h * (self.ratio[6] - 1.), 1]
                        bottom_left = [tlbr[0], tlbr[3] + h * (self.ratio[6] - 1.), 1]
                    else:
                        bottom = [(tlbr[0] + tlbr[2])/2, tlbr[3], 1]
                        bottom_left = [tlbr[0], tlbr[3], 1]
            
            bottom_transformed = self.transform(self.homography_matrix, bottom)
            bottom_transformed = [bottom_transformed[0][0],bottom_transformed[1][0]]
            tracker.tracked_stracks[i].location = [np.array(bottom_transformed), None]

    def transform(self, H, cam_observed_position):
        cam_observed_position = [
            [cam_observed_position[0]],
            [cam_observed_position[1]],
            [1]
        ]
        est_position = np.matmul(H, np.array(cam_observed_position))
        est_position = est_position / est_position[2, :]
        est_position = est_position[:2, :]
        # est_position = np.round(est_position, 0).astype(np.int)

        return est_position