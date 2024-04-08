from sklearn.cluster import AgglomerativeClustering
import cv2
import sys
import os
import numpy as np
sys.path.append('.')
from reids.fastreid.models.model import ReID
from trackers.multicam_tracker.matching import embedding_distance

img_paths = sorted(os.listdir('tools/temp'))
imgs = []
print(img_paths)
features = []

clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')

ReID.initialize(max_batch_size=1)
reid = ReID()

for img_path in img_paths:
    img = cv2.imread(os.path.join('tools/temp', img_path))
    imgs.append(img)
    h,w,_ = img.shape
    bbox = [0,0,w,h]

    feat = reid.run(img, [bbox], 1)
    features.append(feat)

features = np.array(features)
features = np.squeeze(features, 1)
clustering.fit(features)
labels = clustering.labels_

for img, label in zip(img_paths, labels):
    print(f"{img}: {label}")

a_cluster_features = features[labels==0]
b_cluster_features = features[labels==1]

# Mean or One representative (max pose, max area, etc)
# a_cluster_feat = [np.mean(a_cluster_features, axis=0)]
# b_cluster_feat = [np.mean(b_cluster_features, axis=0)]
# emb_dist = embedding_distance(a_cluster_feat, b_cluster_feat) / 2.0
# emb_dist = emb_dist[0][0]
# print(emb_dist)

# emb_dists = embedding_distance(a_cluster_features, b_cluster_features) / 2.0
# emb_dist = np.mean(emb_dists)
# print(emb_dists)
# print(emb_dist)

a_imgs = np.array(imgs)[labels==0]
b_imgs = np.array(imgs)[labels==1]
max_area = 0
max_a_feat = None
for img, feat in zip(a_imgs, a_cluster_features):
    h,w,_ = img.shape
    area = h*w
    if area > max_area:
        max_area = area
        max_a_feat = [feat]
max_area = 0
max_b_feat = None
for img, feat in zip(b_imgs, b_cluster_features):
    h,w,_ = img.shape
    area = h*w
    if area > max_area:
        max_area = area
        max_b_feat = [feat]
emb_dist = embedding_distance(max_a_feat, max_b_feat) / 2.0
emb_dist = emb_dist[0][0]
print(emb_dist)

ReID.finalize()