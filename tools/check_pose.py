import cv2
import sys
sys.path.append('.')
from rtmpose.models.model import RTMPose
import numpy as np

def visualize_kpt(img,
              keypoints,
              scores,
              thr=0.3) -> np.ndarray:
    skeleton = [
        [12, 13], [13, 0], [13, 1], [0, 1], [6, 7], [0, 2], [2, 4], 
        [1, 3], [3, 5], [0, 6], [1, 7], [6, 8], [8, 10], [7, 9], [9, 11]
    ]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    point_color = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

    for kpts, score in zip(keypoints, scores):
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt.astype(np.int32)), 2, palette[color], 2,
                       cv2.LINE_AA)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, tuple(kpts[u].astype(np.int32)),
                         tuple(kpts[v].astype(np.int32)), palette[color], 1,
                         cv2.LINE_AA)

    return img


RTMPose.initialize(max_batch_size=1)
pose = RTMPose()

img_name = '181_0'
# img = cv2.imread(f'tools/overlapped/{img_name}.jpg')
img = cv2.imread('/home/workspace/sample.jpg')
h, w, _ = img.shape
bbox = np.array([[175,0,1725,h]])

pose_result = pose.run({'img': img, 'bboxes': bbox}, 32)
pose_result = np.concatenate([pose_result[0], np.expand_dims(pose_result[1], axis=2)], axis=2)
print(pose_result)
keypoints = [p[:,:2] for p in pose_result]
scores = [p[:,2] for p in pose_result]
img = visualize_kpt(img, keypoints, scores, thr=0.3)

# cv2.imwrite(f'tools/overlapped/{img_name}_pose.jpg', img)
cv2.imwrite('/home/workspace/sample_pose.jpg', img)

RTMPose.finalize()


