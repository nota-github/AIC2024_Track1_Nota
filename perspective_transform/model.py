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
    # def __init__(self, calibration_position, map_size:tuple, ransac_thresh=5.0, maxIters=100000):
    def __init__(self, calibration):
        with open(calibration, 'r') as f:
            matrixes = json.load(f)
        self.camera_projection_matrix_inv = np.array(matrixes['camera projection matrix']).astype(np.float32)
        self.homography_matrix_inv = np.array(matrixes['homography matrix']).astype(np.float32)
        # [pose None=가로길이 3.5배, head, hip or wrist, knee, neck, shoulder, elbow]
        self.ratio = np.array([3.5, 7., 2., 1.6, 7., 5., 3.])
        self.pose_thr = 0.3
        
        # self.calibration_position = calibration_position
        # self.map_size = map_size
        # self.ransac_thresh = ransac_thresh
        # self.pose_thr = 0.3
        # self.maxIters = maxIters
        self.initialize()
    
    def initialize(self):
        # self.homography_matrix, self.mask = self.find_homography(
        #     np.array(self.calibration_position["cam_position"]), 
        #     np.array(self.calibration_position["map_position"])
        # )
        self.homography_matrix = np.linalg.inv(self.homography_matrix_inv)
    
    def run(self, tracker, new_ratio, cam_id=None):
        for i in range(len(tracker.tracked_stracks)):
            tlbr = tracker.tracked_stracks[i].tlbr.tolist()
            pose = tracker.tracked_stracks[i].pose
            self.ratio = self.ratio * 0.9 + new_ratio
            
            
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
            # bottom_left_transformed = self.transform(self.homography_matrix, bottom_left)
            bottom_transformed = [bottom_transformed[0][0],bottom_transformed[1][0]]
            # bottom_transformed = np.maximum(1, bottom_transformed)
            # bottom_transformed = np.minimum((self.map_size[0]-1,self.map_size[1]-1), bottom_transformed)
            # bottom_left_transformed = [bottom_left_transformed[0][0],bottom_left_transformed[1][0]]
            # bottom_left_transformed = np.maximum(1, bottom_left_transformed)
            # bottom_left_transformed = np.minimum((self.map_size[0]-1,self.map_size[1]-1), bottom_left_transformed)
            # confidence = norm.pdf(np.abs([a-b for a,b in zip(bottom_transformed, bottom_left_transformed)]))
            tracker.tracked_stracks[i].location = [np.array(bottom_transformed), None]
            # tracker.tracked_stracks[i].location = [bottom_transformed, confidence]
    
    def find_homography(self, cam_position, map_position):
         # RANSAC Candidates : cv2.USAC_ACCURATE, cv2.USAC_MAGSAC, cv2.RANSAC
        H, status = cv2.findHomography(cam_position, map_position, cv2.USAC_MAGSAC, self.ransac_thresh, maxIters=self.maxIters)
        return H, status

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