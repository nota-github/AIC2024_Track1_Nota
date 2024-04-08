# from ultralytics import YOLO
from detection.models.model import NPNet
from reids.fastreid.models.model import ReID
from rtmpose.models.model import RTMPose

from trackers.botsort.bot_sort import BoTSORT

from trackers.multicam_tracker.cluster_track import MCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

# from mmpose.apis import init_model

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import calibration_position
from tools.utils import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    visualize, update_result_lists_testset, sources, result_paths, map_infos, write_map, cam_ids)

import cv2
import os
import time
import numpy as np
import argparse
import sys
import pdb


def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene):
    # assert len(sources) == len(result_paths[0]), 'length of sources and result_paths is different'
    # detection model initilaize
    # if scene in ['S021', 'S022']:
    #     detection = YOLO('pretrained/yolov8x_aic.pt')
    # else:
    #     detection = YOLO('pretrained/yolov8x6_aic.pt')
    NPNet.initialize()
    detection = NPNet()
    detection.conf_thres = conf_thres
    detection.iou_thres = iou_thres
    classes = detection.classes

    # reid model initilaize
    ReID.initialize(max_batch_size=args['max_batch_size'])
    reid = ReID()

    # pose estimation initialize
    RTMPose.initialize(max_batch_size=args['max_batch_size'])
    pose = RTMPose()
    
    # config_file = './configs/pose/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_hrnet-w32_8xb64-210e_crowdpose-256x192.py'
    # checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192-960be101_20201227.pth'
    # pose = init_model(config_file, checkpoint_file, device='cpu')
    # pose = init_model(config_file, checkpoint_file, device='cuda:0')
    # trackers initialize
    trackers = []
    for i in range(len(sources)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))

    # perspective transform initialize
    calibrations = calibration_position[perspective]
    # perspective_transforms = [PerspectiveTransform(c, map_infos[perspective]['size'], args['ransac_thresh']) for c in calibrations]
    perspective_transforms = [PerspectiveTransform(c) for c in calibrations]
    # for pt in perspective_transforms:
    #     tl, tr, br, bl = [0,0], [1920,0], [1920,1080], [0,1080]
    #     print(f"\nmap tl: ", pt.transform(pt.homography_matrix, tl).tolist())
    #     print(f"map tr: ", pt.transform(pt.homography_matrix, tr).tolist())
    #     print(f"map br: ", pt.transform(pt.homography_matrix, br).tolist())
    #     print(f"map bl: ", pt.transform(pt.homography_matrix, bl).tolist())
 
    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
                            # match_thresh=0.8, map_size=map_infos[perspective]['size'])
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8)
    # mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, map_size=map_infos[perspective]['size'])
    id_distributor = ID_Distributor()

    # get source imgs, video writers
    src_handlers = [get_reader_writer(s) for s in sources]
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format
    # map_img = cv2.imread(map_infos[perspective]['source'])
    map_img = np.full((1280, 720, 3), 255, dtype=np.uint8)  # height, width, 3
    map_writer = cv2.VideoWriter(f'output_videos/{scene}/map_{scene}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 1280))
    # map_writer = cv2.VideoWriter(map_infos[perspective]['savedir'], cv2.VideoWriter_fourcc(*'mp4v'), 30, map_infos[perspective]['size'])

    # total_frames = len(src_handlers[0][0])
    total_frames = max([len(s[0]) for s in src_handlers])
    cur_frame = 1
    stop = False

    while True:
        imgs = []
        start = time.time()

        read_times = []
        det_times = []
        tr_times = []
        
        # first, run trackers each frame independently
        for (img_paths, writer), tracker, perspective_transform, result_list in zip(src_handlers, trackers, perspective_transforms, results_lists):
            # if len(img_paths) == 0 or cur_frame==500:
            if len(img_paths) == 0:
                stop = True
                break
            s = time.time()
            img_path = img_paths.pop(0)
            img = cv2.imread(img_path)
            read_times.append(time.time() - s)
            
            # g = 2.0
            # img = img.astype(np.float64)
            # img = ((img / 255) ** (1 / g)) * 255  # gamma correction
            # img = img.astype(np.uint8)
            
            s = time.time()
            dets = detection.run(img)  # run detection model
            # dets = detection(img, conf=conf_thres, iou=iou_thres, classes=0)[0].boxes.data.cpu().numpy()  # run detection model
            det_times.append(time.time() - s)
            s = time.time()
            online_targets = tracker.update(np.array(dets), img, img_path, reid, pose)  # run tracker
            # online_targets = tracker.update(dets, img, pose)  # run tracker
            tr_times.append(time.time() - s)
            perspective_transform.run(tracker)  # run perspective transform

            # assign global_id to each track for multi-camera tracking
            for t in tracker.tracked_stracks:
                t.t_global_id = id_distributor.assign_id()  # assign temporal global_id
            imgs.append(img)
        # if stop: break
        if stop: break  # 추가 !!!
        
        # second, run multi-camera tracker using above trackers results
        s = time.time()
        groups = clustering.update(trackers, cur_frame, scene)
        mc_tracker.update(trackers, groups)
        clustering.update_using_mctracker(trackers, mc_tracker)

        mct_time = time.time() - s

        r = time.time()
        # if cur_frame % 5 == 0:
        #     mc_tracker.refinement_clusters()
        refine_time = time.time() - s

        latency = time.time() - start

        # update result lists using updated trackers
        update_result_lists_testset(trackers, results_lists, cur_frame, cam_ids, scene)
        
        if args['write_vid']:
            start_write = time.time()
            write_vids(trackers, imgs, src_handlers, latency, pose, _COLORS, mc_tracker, cur_frame)
            map_img = write_map(trackers, map_img, map_writer, _COLORS, mc_tracker, cur_frame)
            print(f'writing time: {(time.time() - start_write):.4f} s')
        
        print(f"video frame ({cur_frame}/{total_frames}) ({latency:.5f} s)")
        print(f"read: {np.sum(read_times):.4f} s / det: {np.sum(det_times):.4f} s / tr + reid + pose: {np.sum(tr_times):.4f} s / mct: {mct_time:.4f} s\n")
        cur_frame += 1
    
    finalize_cams(src_handlers)
    map_writer.release()
    mc_tracker.crop_cluster_people(f'output_cluster/{scene}')

    # save results txt
    write_results_testset(results_lists, result_paths)

    NPNet.finalize()
    ReID.finalize()
    RTMPose.finalize()
    print('Done')


if __name__ == '__main__':
    args = {
        'max_batch_size' : 32,  # maximum input batch size of reid model
        'track_buffer' : 150,  # the frames for keep lost tracks
        'with_reid' : True,  # whether to use reid model's out feature map at first association
        'sct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do single-cam tracking
        'sct_euclidean_thresh' : 0.1,  # threshold of euclidean distance when do single-cam tracking

        'clt_appearance_thresh' : 0.35,  # threshold of appearance feature cosine distance when do multi-cam clustering
        'clt_euclidean_thresh' : 0.3,  # threshold of euclidean distance when do multi-cam clustering

        'mct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do cluster tracking (not important)

        'ransac_thresh' : 10,  # threshold of ransac when find homography matrix 
        'frame_rate' : 30,  # your video(camera)'s fps
        'write_vid' : True,  # write result to video
        }

    # val
    scenes = ['scene_072']
    for scene in scenes:
        run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)

    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S003'], result_paths=result_paths['S003'], perspective='S003', cam_ids=cam_ids['S003'], scene='S003')
    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S009'], result_paths=result_paths['S009'], perspective='S009', cam_ids=cam_ids['S009'], scene='S009')
    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S014'], result_paths=result_paths['S014'], perspective='S014', cam_ids=cam_ids['S014'], scene='S014')
    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S018'], result_paths=result_paths['S018'], perspective='S018', cam_ids=cam_ids['S018'], scene='S018')
    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S021'], result_paths=result_paths['S021'], perspective='S021', cam_ids=cam_ids['S021'], scene='S021')
    # run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S022'], result_paths=result_paths['S022'], perspective='S022', cam_ids=cam_ids['S022'], scene='S022')