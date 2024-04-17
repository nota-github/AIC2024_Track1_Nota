from ultralytics import YOLO
from mmpose.apis import init_model

from trackers.botsort.bot_sort import BoTSORT
from trackers.multicam_tracker.cluster_track import MCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import calibration_position

from tools.utils import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    update_result_lists_testset, sources, result_paths, cam_ids)

import cv2
import os
import time
import numpy as np
import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="Run Online MTPC System")
    parser.add_argument("-s", "--scene", type=str, default=None, help="scene name to inference")
    return parser.parse_args()

def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene):
    # detection model initilaize
    if int(scene.split('_')[1]) in range(61,81):
        detection = YOLO('pretrained/yolov8x_aic.pt')
    else:
        detection = YOLO('yolov8x.pt')

    # pose estimation initialize
    config_file = '/mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(config_file, checkpoint_file, device='cuda:0')

    # trackers initialize
    trackers = []
    for i in range(len(sources)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))

    # perspective transform initialize
    calibrations = calibration_position[perspective]
    perspective_transforms = [PerspectiveTransform(c) for c in calibrations]
 
    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # get source imgs, video writers
    src_handlers = [get_reader_writer(s) for s in sources]
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format

    total_frames = max([len(s[0]) for s in src_handlers])
    cur_frame = 1
    stop = False

    while True:
        imgs = []
        start = time.time()
        
        # first, run trackers each frame independently
        for (img_paths, writer), tracker, perspective_transform, result_list in zip(src_handlers, trackers, perspective_transforms, results_lists):
            # if len(img_paths) == 0 or cur_frame==30:
            if len(img_paths) == 0:
                stop = True
                break
            img_path = img_paths.pop(0)
            img = cv2.imread(img_path)
            
            # run detection model
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=0, verbose=False)[0].boxes.data.cpu().numpy()

            # run tracker
            online_targets, new_ratio = tracker.update(dets, img, img_path, pose)  # run tracker

            # run perspective transform
            perspective_transform.run(tracker, new_ratio)

            # assign temporal global_id to each track for multi-camera tracking
            for t in tracker.tracked_stracks:
                t.t_global_id = id_distributor.assign_id()
            imgs.append(img)
        if stop: break
        
        # second, run multi-camera tracker using above trackers results
        groups = clustering.update(trackers, cur_frame, scene)
        mc_tracker.update(trackers, groups)
        clustering.update_using_mctracker(trackers, mc_tracker)

        # third, run cluster self-refinements
        if cur_frame % 5 == 0:
            mc_tracker.refinement_clusters()

        # update result lists using updated trackers
        update_result_lists_testset(trackers, results_lists, cur_frame, cam_ids, scene)
        
        if args['write_vid']:
            write_vids(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame)
        
        print(f"video frame ({cur_frame}/{total_frames})")
        cur_frame += 1
    
    finalize_cams(src_handlers)

    # save results txt
    write_results_testset(results_lists, result_paths)
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

        'frame_rate' : 30,  # your video(camera)'s fps
        'write_vid' : False,  # write result to video
        }

    scene = make_parser().scene

    if scene is not None:
        run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)
        
    else:
        # run each scene sequentially
        scenes = [
                'scene_061', 'scene_062', 'scene_063', 'scene_064', 'scene_065', 'scene_066', 'scene_067', 'scene_068', 'scene_069', 'scene_070',
                'scene_071', 'scene_072', 'scene_073', 'scene_074', 'scene_075', 'scene_076', 'scene_077', 'scene_078', 'scene_079', 'scene_080',
                'scene_081', 'scene_082', 'scene_083', 'scene_084', 'scene_085', 'scene_086', 'scene_087', 'scene_088', 'scene_089', 'scene_090',
                ]
        for scene in scenes:
            run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)
