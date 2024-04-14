import numpy as np
import os
import cv2
from collections import deque
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

from . import matching
from .basetrack import BaseTrack, TrackState

from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import sys


class MTrack(BaseTrack):
    def __init__(self, global_id, centroid, feat, pose, img_path, tlbr, coords, min_hits=30, scene=None, feat_history=50):
        self.is_activated = False
        self.centroid = centroid
        self.global_id = global_id

        self.smooth_feat = None
        self.curr_feat = None
        self.curr_pose = None
        self.curr_coords = coords

        self.img_paths = img_path
        self.tlbrs = tlbr
        self.path_tlbr = {}
        self.poses = deque([], maxlen=10)
        self.features = deque([], maxlen=10)
        self.pose_thresh = 10

        if feat is not None:
            self.update_features(feat, pose, img_path, tlbr, coords)
        self.alpha = 0.9
        self.min_hits = min_hits

    def compare_poses(self, curr_pose):
        for pose in self.poses:
            pose1 = np.array(pose['keypoints'][:, :2])
            pose2 = np.array(curr_pose['keypoints'][:, :2])
            _, _, distance = procrustes(pose1, pose2)
            if distance < 1e-8:
                return True
        return False
    
    def update_features(self, features, poses, img_paths, tlbrs, coords):
        self.curr_feat = features
        self.curr_pose = poses
        self.curr_coords = coords

        for feat, pose, img_path, tlbr in zip(features, poses, img_paths, tlbrs):
            if pose is None: continue
            if len(self.features) == self.features.maxlen: return
            num_point = sum(pose['keypoints'][:, 2] > 0.5)
            if num_point >= self.pose_thresh:
                """ Procrustes Analysis """
                if img_path in self.path_tlbr or self.compare_poses(pose): continue
                self.features.append(feat)
                self.path_tlbr[img_path] = tlbr
                self.poses.append(pose)
        
        if len(self.features) == 0:
            max_num, max_feat = 0, None
            for feat, pose, img_path, tlbr in zip(features, poses, img_paths, tlbrs):
                if pose is None: continue
                if sum(pose['keypoints'][:, 2] > 0.5) > max_num:
                    max_num = sum(pose['keypoints'][:, 2] > 0.5)
                    max_feat = feat
                    max_path = img_path
                    max_tlbr = tlbr
                    max_pose = pose
            
            if max_num > 0:
                if max_path in self.path_tlbr: return
                self.features.append(max_feat)
                self.path_tlbr[max_path] = max_tlbr
                self.poses.append(pose)
            else:
                # self.features.extend(features)
                for feat, img_path, tlbr in zip(features, poses, img_paths, tlbrs):
                    if img_path in self.path_tlbr: continue
                    self.features.append(feat)
                    self.path_tlbr[img_path] = tlbr
                    self.poses.append(pose)
                
    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.curr_pose, new_track.img_paths, new_track.tlbrs, new_track.curr_coords)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.curr_pose, new_track.img_paths, new_track.tlbrs, new_track.curr_coords)

        self.state = TrackState.Tracked
        if self.tracklet_len > self.min_hits:
            self.is_activated = True


class MCTracker:
    def __init__(self, appearance_thresh=0.8, euc_thresh=0.5, match_thresh=0.8, scene=None, max_time_lost=18000, min_hits=10):
        self.tracked_mtracks = []  # type: list[MTrack]
        self.lost_mtracks = []  # type: list[MTrack]
        self.removed_mtracks = []  # type: list[MTrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        # self.match_thresh = 0.99
        self.max_len = 1

        self.clustering = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')
        # self.clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='ward')

        if int(scene.split('_')[1]) in range(61, 71):
            self.emb_thresh = 0.30
            self.euc_thresh = 1.5
            self.refine_1st_emb = 0.325
            self.refine_2nd_emb = 0.325
            self.refine_2nd_euc = 1
            self.refine_3rd_emb = 0.30

        elif int(scene.split('_')[1]) in range(71, 81):
            self.emb_thresh = 0.325
            self.euc_thresh = 1.5
            self.refine_1st_emb = 0.325
            self.refine_2nd_emb = 0.35
            self.refine_2nd_euc = 1
            self.refine_3rd_emb = 0.325

        elif int(scene.split('_')[1]) in range(81, 91):
            self.emb_thresh = 0.35
            self.euc_thresh = 1.5
            self.refine_1st_emb = 0.25
            self.refine_2nd_emb = 0.325
            self.refine_2nd_euc = 1
            self.refine_3rd_emb = 0.30

        else:
            print('Not Test Set Scene')
            raise
    
    def update(self, trackers, groups, scene=None):
        self.frame_id += 1
        activated_mtracks = []
        refind_mtracks = []
        lost_mtracks = []
        removed_mtracks = []

        if len(groups):  # group type: array[[t_groud_id, features, centroid(location)], ...]
            global_ids = groups[:, 0]
            features = groups[:, 1]
            centroids = groups[:, 2]
            poses = groups[:, 3]

            paths = groups[:, 4]
            tlbrs = groups[:, 5]
            coords = groups[:, 6]
        else:
            global_ids = []
            features = []
            centroids = []
            poses = []

            paths = []
            tlbrs = []
            coords = []
        
        if len(centroids) > 0:
            new_groups = [MTrack(g, c, f, p, ph, t, cd, self.min_hits, scene) for (g, c, f, p, ph, t, cd) in zip(global_ids, centroids, features, poses, paths, tlbrs, coords)]
        else:
            new_groups = []

        ''' Step 1: Add newly detected groups to tracked_mtracks '''
        unconfirmed = []
        tracked_mtracks = []  # type: list[MTrack]
        for track in self.tracked_mtracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_mtracks.append(track)
        
        ''' Step 2: First Association with tracked mtracks '''
        mtrack_pool = joint_mtracks(tracked_mtracks, self.lost_mtracks)
        exist_features = [feat for m in tracked_mtracks for feat in list(m.features)]
        lengths_exists = [len(m.features) for m in tracked_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]
        exist_centroids = [m.centroid for m in tracked_mtracks]
        new_centroids = [g.centroid for g in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        elif True:
            if self.frame_id % 10 == 0:
                rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
                emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
                dists = emb_dists
            else:
                rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
                emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
                euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len
                norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
                norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))
                dists = 0.5 * norm_euc_dists + 0.5 * norm_emb_dists
                dists[euc_dists > 1] = 1.0
        else:
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len

            dists = emb_dists

        matches, u_exist, u_new = matching.linear_assignment(dists, thresh=0.999)

        for iexist, inew in matches:
            exist = tracked_mtracks[iexist]
            new = new_groups[inew]
            if exist.state == TrackState.Tracked:
                exist.update(new, self.frame_id)
                activated_mtracks.append(exist)
            else:
                exist.re_activate(new, self.frame_id, new_id=False)
                refind_mtracks.append(exist)
        
        for it in u_exist:
            track = tracked_mtracks[it]
            if not track.state == TrackState.Lost and (not track.state == TrackState.Removed):
                track.mark_lost()
                lost_mtracks.append(track)

        ''' Step 3: Second association with lost mtracks '''
        new_groups = [new_groups[i] for i in u_new]

        lost_features = [feat for m in self.lost_mtracks for feat in list(m.features)]
        lengths_lost = [len(m.features) for m in self.lost_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]

        shape = (len(lengths_lost), len(lengths_new))
        if 0 in shape:
            emb_dists = np.empty((len(lengths_lost), len(lengths_new)))
        else:
            rerank_dists = matching.embedding_distance(lost_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_lost, lengths_new, shape, normalize=False)

        dists = emb_dists

        matches, u_lost, u_new = matching.linear_assignment(dists, thresh=self.emb_thresh)

        for ilost, inew in matches:
            lost = self.lost_mtracks[ilost]
            new = new_groups[inew]
            lost.re_activate(new, self.frame_id, new_id=False)
            refind_mtracks.append(lost)

        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame '''
        new_groups = [new_groups[i] for i in u_new]

        exist_centroids = [m.centroid for m in unconfirmed]
        new_centroids = [g.centroid for g in new_groups]

        exist_features = [feat for m in unconfirmed for feat in list(m.features)]
        lengths_exists = [len(m.features) for m in unconfirmed]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        else:
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len

            norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
            norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))
            dists = 0.5 * norm_euc_dists + 0.5 * norm_emb_dists

            if shape == (1,1): dists = emb_dists
            dists[euc_dists > self.euc_thresh] = 1.0

        matches, u_unconfirmed, u_new = matching.linear_assignment(dists, thresh=0.999)
        for iexist, inew in matches:
            unconfirmed[iexist].update(new_groups[inew], self.frame_id)
            activated_mtracks.append(unconfirmed[iexist])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_mtracks.append(track)

        """ Step 5: Init new mtracks """
        for inew in u_new:
            track = new_groups[inew]
            # if track.score < self.new_track_thresh:
            #     continue
            track.activate(self.frame_id)
            activated_mtracks.append(track)
        
        """ Step 6: Update state """
        for track in self.lost_mtracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_mtracks.append(track)

        """ Merge """
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == TrackState.Tracked]
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, activated_mtracks)
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks.extend(lost_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, removed_mtracks)
        # self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.removed_mtracks)
        # self.removed_mtracks.extend(removed_mtracks)

        # self.tracked_mtracks, self.lost_mtracks = remove_duplicate_mtracks(self.tracked_mtracks, self.lost_mtracks)

        output_mtracks = [track for track in self.tracked_mtracks if track.is_activated]
        unconfirmed_mtracks = [track for track in self.tracked_mtracks if not track.is_activated]
        print(f'tracking ids: {[m.track_id for m in output_mtracks]}')
        print(f'unconfirmed_tracks ids: {[[m.track_id, m.tracklet_len]for m in unconfirmed_mtracks]}')
        print(f'lost_tracks ids: {[m.track_id for m in self.lost_mtracks]}')

        for mtrack in self.tracked_mtracks:
            if mtrack.start_frame == 1 or (not track.is_activated):
                pass
            elif mtrack.frame_id - mtrack.start_frame >= 30:
                mtrack.start_frame = 1

    def get_max_area_feature(self, features, keys, path_tlbr):
        max_area = 0
        max_feat = None
        for feat, key in zip(features, keys):
            x1, y1, x2, y2 = path_tlbr[key]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_feat = feat
        return [max_feat]

    def refinement_clusters(self):
        tracked_mtracks = [track for track in self.tracked_mtracks if track.is_activated]
        unconfirmed_mtracks = [track for track in self.tracked_mtracks if not track.is_activated]
        lost_mtracks = self.lost_mtracks
        mtracks = tracked_mtracks + unconfirmed_mtracks + lost_mtracks

        ''' Step 1: Find cluster(tracked_mtracks) consists of two people '''
        for mtrack in mtracks:
            features = np.array(mtrack.features)
            if features.shape[0] == 1: continue
            self.clustering.fit(features)
            labels = self.clustering.labels_

            path_tlbr = mtrack.path_tlbr.copy()
            a_path_keys = np.array(list(path_tlbr.keys()))[labels==0]
            b_path_keys = np.array(list(path_tlbr.keys()))[labels==1]

            a_cluster_features = features[labels==0]
            b_cluster_features = features[labels==1]

            a_max_area_feat = self.get_max_area_feature(a_cluster_features, a_path_keys, path_tlbr)
            b_max_area_feat = self.get_max_area_feature(b_cluster_features, b_path_keys, path_tlbr)

            emb_dist = matching.embedding_distance(a_max_area_feat, b_max_area_feat) / 2.0
            emb_dist = emb_dist[0][0]

            # cluster is consist of two people -> regard latest added person's features
            if emb_dist > self.refine_1st_emb:
                print(f'refine step1: cluster {mtrack.track_id} is consist of two people ({sum(labels == 0)} + {sum(labels == 1)}) / dist: {emb_dist:.4f}')
                a_mean_ind = np.mean(np.where(labels == 0)[0])
                b_mean_ind = np.mean(np.where(labels == 1)[0])
                
                path_tlbr = mtrack.path_tlbr.copy()
                mtrack.path_tlbr = {}
                mtrack.features = deque([], maxlen=10)
                if a_mean_ind <= b_mean_ind:
                    # mtrack.features = deque([], maxlen=len(a_cluster_features))
                    mtrack.features.extend(a_cluster_features)
                    path_keys = np.array(list(path_tlbr.keys()))[labels==0]
                else:
                    # mtrack.features = deque([], maxlen=len(b_cluster_features))
                    mtrack.features.extend(b_cluster_features)
                    path_keys = np.array(list(path_tlbr.keys()))[labels==1]
                for p in path_keys:
                    mtrack.path_tlbr[p] = path_tlbr[p]
        
        ''' Step 2: Find clusters(tracked_mtracks) consists of same person '''
        tracked_mtracks = [track for track in self.tracked_mtracks if track.is_activated]
        cluster_pairs = list(combinations(tracked_mtracks, 2))
        removed_mtracks = []

        for a_cluster, b_cluster in cluster_pairs:
            if a_cluster.state == TrackState.Removed or b_cluster.state == TrackState.Removed: continue
            a_cluster_features = list(a_cluster.features)
            b_cluster_features = list(b_cluster.features)
            emb_dists = matching.embedding_distance(a_cluster_features, b_cluster_features) / 2.0
            emb_dist = np.mean(emb_dists)

            a_cluster_centroid = [a_cluster.centroid]
            b_cluster_centroid = [b_cluster.centroid]
            euc_dist = matching.euclidean_distance(a_cluster_centroid, b_cluster_centroid)[0][0]

            # both clusters are consist of same person -> remove latest created cluster
            if emb_dist < self.refine_2nd_emb and euc_dist < self.refine_2nd_euc:
                if a_cluster.start_frame == 1 and b_cluster.start_frame == 1 and self.frame_id > 10: continue
                print(f'refine step2: cluster {a_cluster.track_id} and cluster {b_cluster.track_id} are same person')
                if a_cluster.track_id <= b_cluster.track_id:
                    b_cluster.mark_removed()
                    removed_mtracks.append(b_cluster)
                else:
                    a_cluster.mark_removed()
                    removed_mtracks.append(a_cluster)

        self.tracked_mtracks = [t for t in tracked_mtracks if t.state == TrackState.Tracked]

        ''' Step 3: Compare with lost clusters to remove not new cluster'''
        tracked_mtracks = [track for track in self.tracked_mtracks if track.is_activated]
        lost_mtracks = self.lost_mtracks
        refind_mtracks = []

        for tracked_cluster in tracked_mtracks:
            if tracked_cluster.state == TrackState.Removed: continue
            for lost_cluster in lost_mtracks:
                if lost_cluster.state == TrackState.Tracked: continue
                tracked_cluster_feats = list(tracked_cluster.features)
                lost_cluster_feats = list(lost_cluster.features)
                
                emb_dists = matching.embedding_distance(tracked_cluster_feats, lost_cluster_feats) / 2.0
                emb_dist = np.mean(emb_dists)

                if emb_dist < self.refine_3rd_emb and lost_cluster.track_id < tracked_cluster.track_id and tracked_cluster.start_frame != 1:
                    print(f'refine step3: cluster {tracked_cluster.track_id} is removed (matched with lost cluster {lost_cluster.track_id})')
                    lost_cluster.re_activate(tracked_cluster, self.frame_id, new_id=False)
                    refind_mtracks.append(lost_cluster)
                    tracked_cluster.mark_removed()
                    removed_mtracks.append(tracked_cluster)
                    break

        """ Merge """
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == TrackState.Tracked]
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, unconfirmed_mtracks)
        # self.removed_mtracks.extend(removed_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, removed_mtracks)
        # self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.removed_mtracks)


def grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=True):
    emb_dists = np.zeros(shape, dtype=np.float)
    total_sum = np.sum(rerank_dists)
    num = 0
    ratio = 0
    for i, len_e in enumerate(lengths_exists):
        for j, len_n in enumerate(lengths_new):
            start_x = sum(lengths_exists[:i])
            end_x = start_x + lengths_exists[i]
            start_y = sum(lengths_new[:j])
            end_y = start_y + lengths_new[j]
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum
            if shape == (1,1):
                emb_dists[i,j] = np.mean(rerank_dists[start_x:end_x, start_y:end_y]) 
            else:
                # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum / (len_e*len_n)  # origin
                emb_dists[i,j] = np.mean(rerank_dists[start_x:end_x, start_y:end_y])
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) 
    max_val = np.max(emb_dists)
    min_val = np.min(emb_dists)
    if shape != (1, 1) and normalize:
        emb_dists = (emb_dists - min_val) / (max_val - min_val)
    return emb_dists

def joint_mtracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_mtracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


# def remove_duplicate_mtracks(stracksa, stracksb):
#     pdist = matching.iou_distance(stracksa, stracksb)
#     pairs = np.where(pdist < 0.15)
#     dupa, dupb = list(), list()
#     for p, q in zip(*pairs):
#         timep = stracksa[p].frame_id - stracksa[p].start_frame
#         timeq = stracksb[q].frame_id - stracksb[q].start_frame
#         if timep > timeq:
#             dupb.append(q)
#         else:
#             dupa.append(p)
#     resa = [t for i, t in enumerate(stracksa) if not i in dupa]
#     resb = [t for i, t in enumerate(stracksb) if not i in dupb]
#     return resa, resb