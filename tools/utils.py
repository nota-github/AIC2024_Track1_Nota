import numpy as np
from pathlib import Path
import cv2
import os
import pickle
import json


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

sources = {
        # Val
        'scene_041': sorted([os.path.join('/workspace/frames/val/scene_041', p) for p in os.listdir('/workspace/frames/val/scene_041')]),
        'scene_042': sorted([os.path.join('/workspace/frames/val/scene_042', p) for p in os.listdir('/workspace/frames/val/scene_042')]),
        'scene_043': sorted([os.path.join('/workspace/frames/val/scene_043', p) for p in os.listdir('/workspace/frames/val/scene_043')]),
        'scene_044': sorted([os.path.join('/workspace/frames/val/scene_044', p) for p in os.listdir('/workspace/frames/val/scene_044')]),
        'scene_045': sorted([os.path.join('/workspace/frames/val/scene_045', p) for p in os.listdir('/workspace/frames/val/scene_045')]),
        'scene_046': sorted([os.path.join('/workspace/frames/val/scene_046', p) for p in os.listdir('/workspace/frames/val/scene_046')]),
        'scene_047': sorted([os.path.join('/workspace/frames/val/scene_047', p) for p in os.listdir('/workspace/frames/val/scene_047')]),
        'scene_048': sorted([os.path.join('/workspace/frames/val/scene_048', p) for p in os.listdir('/workspace/frames/val/scene_048')]),
        'scene_049': sorted([os.path.join('/workspace/frames/val/scene_049', p) for p in os.listdir('/workspace/frames/val/scene_049')]),
        'scene_050': sorted([os.path.join('/workspace/frames/val/scene_050', p) for p in os.listdir('/workspace/frames/val/scene_050')]),
        'scene_051': sorted([os.path.join('/workspace/frames/val/scene_051', p) for p in os.listdir('/workspace/frames/val/scene_051')]),
        'scene_052': sorted([os.path.join('/workspace/frames/val/scene_052', p) for p in os.listdir('/workspace/frames/val/scene_052')]),
        'scene_053': sorted([os.path.join('/workspace/frames/val/scene_053', p) for p in os.listdir('/workspace/frames/val/scene_053')]),
        'scene_054': sorted([os.path.join('/workspace/frames/val/scene_054', p) for p in os.listdir('/workspace/frames/val/scene_054')]),
        'scene_055': sorted([os.path.join('/workspace/frames/val/scene_055', p) for p in os.listdir('/workspace/frames/val/scene_055')]),
        'scene_056': sorted([os.path.join('/workspace/frames/val/scene_056', p) for p in os.listdir('/workspace/frames/val/scene_056')]),
        'scene_057': sorted([os.path.join('/workspace/frames/val/scene_057', p) for p in os.listdir('/workspace/frames/val/scene_057')]),
        'scene_058': sorted([os.path.join('/workspace/frames/val/scene_058', p) for p in os.listdir('/workspace/frames/val/scene_058')]),
        'scene_059': sorted([os.path.join('/workspace/frames/val/scene_059', p) for p in os.listdir('/workspace/frames/val/scene_059')]),
        'scene_060': sorted([os.path.join('/workspace/frames/val/scene_060', p) for p in os.listdir('/workspace/frames/val/scene_060')]),

        # Test
        'scene_061': sorted([os.path.join('/workspace/frames/test/scene_061', p) for p in os.listdir('/workspace/frames/test/scene_061')]),
        'scene_062': sorted([os.path.join('/workspace/frames/test/scene_062', p) for p in os.listdir('/workspace/frames/test/scene_062')]),
        'scene_063': sorted([os.path.join('/workspace/frames/test/scene_063', p) for p in os.listdir('/workspace/frames/test/scene_063')]),
        'scene_064': sorted([os.path.join('/workspace/frames/test/scene_064', p) for p in os.listdir('/workspace/frames/test/scene_064')]),
        'scene_065': sorted([os.path.join('/workspace/frames/test/scene_065', p) for p in os.listdir('/workspace/frames/test/scene_065')]),
        'scene_066': sorted([os.path.join('/workspace/frames/test/scene_066', p) for p in os.listdir('/workspace/frames/test/scene_066')]),
        'scene_067': sorted([os.path.join('/workspace/frames/test/scene_067', p) for p in os.listdir('/workspace/frames/test/scene_067')]),
        'scene_068': sorted([os.path.join('/workspace/frames/test/scene_068', p) for p in os.listdir('/workspace/frames/test/scene_068')]),
        'scene_069': sorted([os.path.join('/workspace/frames/test/scene_069', p) for p in os.listdir('/workspace/frames/test/scene_069')]),
        'scene_070': sorted([os.path.join('/workspace/frames/test/scene_070', p) for p in os.listdir('/workspace/frames/test/scene_070')]),
        'scene_071': sorted([os.path.join('/workspace/frames/test/scene_071', p) for p in os.listdir('/workspace/frames/test/scene_071')]),
        'scene_072': sorted([os.path.join('/workspace/frames/test/scene_072', p) for p in os.listdir('/workspace/frames/test/scene_072')]),
        'scene_073': sorted([os.path.join('/workspace/frames/test/scene_073', p) for p in os.listdir('/workspace/frames/test/scene_073')]),
        'scene_074': sorted([os.path.join('/workspace/frames/test/scene_074', p) for p in os.listdir('/workspace/frames/test/scene_074')]),
        'scene_075': sorted([os.path.join('/workspace/frames/test/scene_075', p) for p in os.listdir('/workspace/frames/test/scene_075')]),
        'scene_076': sorted([os.path.join('/workspace/frames/test/scene_076', p) for p in os.listdir('/workspace/frames/test/scene_076')]),
        'scene_077': sorted([os.path.join('/workspace/frames/test/scene_077', p) for p in os.listdir('/workspace/frames/test/scene_077')]),
        'scene_078': sorted([os.path.join('/workspace/frames/test/scene_078', p) for p in os.listdir('/workspace/frames/test/scene_078')]),
        'scene_079': sorted([os.path.join('/workspace/frames/test/scene_079', p) for p in os.listdir('/workspace/frames/test/scene_079')]),
        'scene_080': sorted([os.path.join('/workspace/frames/test/scene_080', p) for p in os.listdir('/workspace/frames/test/scene_080')]),
        'scene_081': sorted([os.path.join('/workspace/frames/test/scene_081', p) for p in os.listdir('/workspace/frames/test/scene_081')]),
        'scene_082': sorted([os.path.join('/workspace/frames/test/scene_082', p) for p in os.listdir('/workspace/frames/test/scene_082')]),
        'scene_083': sorted([os.path.join('/workspace/frames/test/scene_083', p) for p in os.listdir('/workspace/frames/test/scene_083')]),
        'scene_084': sorted([os.path.join('/workspace/frames/test/scene_084', p) for p in os.listdir('/workspace/frames/test/scene_084')]),
        'scene_085': sorted([os.path.join('/workspace/frames/test/scene_085', p) for p in os.listdir('/workspace/frames/test/scene_085')]),
        'scene_086': sorted([os.path.join('/workspace/frames/test/scene_086', p) for p in os.listdir('/workspace/frames/test/scene_086')]),
        'scene_087': sorted([os.path.join('/workspace/frames/test/scene_087', p) for p in os.listdir('/workspace/frames/test/scene_087')]),
        'scene_088': sorted([os.path.join('/workspace/frames/test/scene_088', p) for p in os.listdir('/workspace/frames/test/scene_088')]),
        'scene_089': sorted([os.path.join('/workspace/frames/test/scene_089', p) for p in os.listdir('/workspace/frames/test/scene_089')]),
        'scene_090': sorted([os.path.join('/workspace/frames/test/scene_090', p) for p in os.listdir('/workspace/frames/test/scene_090')]),
        }

result_paths = {
    # Val
    'scene_041': './results/scene_041.txt',
    'scene_042': './results/scene_042.txt',
    'scene_043': './results/scene_043.txt',
    'scene_044': './results/scene_044.txt',
    'scene_045': './results/scene_045.txt',
    'scene_046': './results/scene_046.txt',
    'scene_047': './results/scene_047.txt',
    'scene_048': './results/scene_048.txt',
    'scene_049': './results/scene_049.txt',
    'scene_050': './results/scene_050.txt',
    'scene_051': './results/scene_051.txt',
    'scene_052': './results/scene_052.txt',
    'scene_053': './results/scene_053.txt',
    'scene_054': './results/scene_054.txt',
    'scene_055': './results/scene_055.txt',
    'scene_056': './results/scene_056.txt',
    'scene_057': './results/scene_057.txt',
    'scene_058': './results/scene_058.txt',
    'scene_059': './results/scene_059.txt',
    'scene_060': './results/scene_060.txt',

    # Test
    'scene_061': './results/scene_061.txt',
    'scene_062': './results/scene_062.txt',
    'scene_063': './results/scene_063.txt',
    'scene_064': './results/scene_064.txt',
    'scene_065': './results/scene_065.txt',
    'scene_066': './results/scene_066.txt',
    'scene_067': './results/scene_067.txt',
    'scene_068': './results/scene_068.txt',
    'scene_069': './results/scene_069.txt',
    'scene_070': './results/scene_070.txt',
    'scene_071': './results/scene_071.txt',
    'scene_072': './results/scene_072.txt',
    'scene_073': './results/scene_073.txt',
    'scene_074': './results/scene_074.txt',
    'scene_075': './results/scene_075.txt',
    'scene_076': './results/scene_076.txt',
    'scene_077': './results/scene_077.txt',
    'scene_078': './results/scene_078.txt',
    'scene_079': './results/scene_079.txt',
    'scene_080': './results/scene_080.txt',
    'scene_081': './results/scene_081.txt',
    'scene_082': './results/scene_082.txt',
    'scene_083': './results/scene_083.txt',
    'scene_084': './results/scene_084.txt',
    'scene_085': './results/scene_085.txt',
    'scene_086': './results/scene_086.txt',
    'scene_087': './results/scene_087.txt',
    'scene_088': './results/scene_088.txt',
    'scene_089': './results/scene_089.txt',
    'scene_090': './results/scene_090.txt',
    }

cam_ids = {
    # Val
    'scene_041': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_041') if cam.startswith('camera_')]),
    'scene_042': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_042') if cam.startswith('camera_')]),
    'scene_043': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_043') if cam.startswith('camera_')]),
    'scene_044': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_044') if cam.startswith('camera_')]),
    'scene_045': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_045') if cam.startswith('camera_')]),
    'scene_046': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_046') if cam.startswith('camera_')]),
    'scene_047': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_047') if cam.startswith('camera_')]),
    'scene_048': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_048') if cam.startswith('camera_')]),
    'scene_049': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_049') if cam.startswith('camera_')]),
    'scene_050': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_050') if cam.startswith('camera_')]),
    'scene_051': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_051') if cam.startswith('camera_')]),
    'scene_052': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_052') if cam.startswith('camera_')]),
    'scene_053': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_053') if cam.startswith('camera_')]),
    'scene_054': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_054') if cam.startswith('camera_')]),
    'scene_055': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_055') if cam.startswith('camera_')]),
    'scene_056': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_056') if cam.startswith('camera_')]),
    'scene_057': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_057') if cam.startswith('camera_')]),
    'scene_058': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_058') if cam.startswith('camera_')]),
    'scene_059': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_059') if cam.startswith('camera_')]),
    'scene_060': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_060') if cam.startswith('camera_')]),

    # Test
    'scene_061': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_061') if cam.startswith('camera_')]),
    'scene_062': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_062') if cam.startswith('camera_')]),
    'scene_063': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_063') if cam.startswith('camera_')]),
    'scene_064': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_064') if cam.startswith('camera_')]),
    'scene_065': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_065') if cam.startswith('camera_')]),
    'scene_066': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_066') if cam.startswith('camera_')]),
    'scene_067': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_067') if cam.startswith('camera_')]),
    'scene_068': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_068') if cam.startswith('camera_')]),
    'scene_069': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_069') if cam.startswith('camera_')]),
    'scene_070': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_070') if cam.startswith('camera_')]),
    'scene_071': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_071') if cam.startswith('camera_')]),
    'scene_072': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_072') if cam.startswith('camera_')]),
    'scene_073': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_073') if cam.startswith('camera_')]),
    'scene_074': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_074') if cam.startswith('camera_')]),
    'scene_075': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_075') if cam.startswith('camera_')]),
    'scene_076': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_076') if cam.startswith('camera_')]),
    'scene_077': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_077') if cam.startswith('camera_')]),
    'scene_078': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_078') if cam.startswith('camera_')]),
    'scene_079': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_079') if cam.startswith('camera_')]),
    'scene_080': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_080') if cam.startswith('camera_')]),
    'scene_081': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_081') if cam.startswith('camera_')]),
    'scene_082': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_082') if cam.startswith('camera_')]),
    'scene_083': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_083') if cam.startswith('camera_')]),
    'scene_084': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_084') if cam.startswith('camera_')]),
    'scene_085': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_085') if cam.startswith('camera_')]),
    'scene_086': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_086') if cam.startswith('camera_')]),
    'scene_087': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_087') if cam.startswith('camera_')]),
    'scene_088': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_088') if cam.startswith('camera_')]),
    'scene_089': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_089') if cam.startswith('camera_')]),
    'scene_090': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_090') if cam.startswith('camera_')]),
}

def get_reader_writer(source):
    src_paths = sorted(os.listdir(source),  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    src_paths = [os.path.join(source, s) for s in src_paths]

    fps = 30
    wi, he = 1920, 1080
    os.makedirs('output_videos/' + source.split('/')[-2], exist_ok=True)
    # dst = 'output_videos/' + source.replace('/','').replace('.','') + '.mp4'
    dst = f"output_videos/{source.split('/')[-2]}/" + source.split('/')[-3] + '_' + source.split('/')[-2] + source.split('/')[-1] + '.mp4'
    video_writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wi, he))
    print(f"{source}'s total frames: {len(src_paths)}")

    return [src_paths, video_writer]

def finalize_cams(src_handlers):
    for s, w in src_handlers:
        w.release()
        print(f"{w} released")

def write_vids(trackers, imgs, src_handlers, pose, colors, mc_tracker, cur_frame=0):
    writers = [w for s, w in src_handlers]
    gid_2_lenfeats = {}
    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)

    for tracker, img, w in zip(trackers, imgs, writers):
        outputs = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        img = visualize(outputs, img, colors, pose, pose_result, cur_frame)
        w.write(img)

def write_results_testset(result_lists, result_path):
    dst_folder = str(Path(result_path).parent)
    os.makedirs(dst_folder, exist_ok=True)
    # write multicam results
    with open(result_path, 'w') as f:
        print(result_path)
        for result in result_lists:
            for r in result:
                t, l, w, h = r['tlwh']
                xworld, yworld = r['2d_coord']
                row = [r['cam_id'], r['track_id'], r['frame_id'], int(t), int(l), int(w), int(h), float(xworld), float(yworld)]
                row = " ".join([str(r) for r in row]) + '\n'
                # row = " ".join(row)
                f.write(row)

def update_result_lists_testset(trackers, result_lists, frame_id, cam_ids, scene):
    results_frame = [[] for i in range(len(result_lists))]
    results_frame_feat = []
    for tracker, result_frame, result_list, cam_id in zip(trackers, results_frame, result_lists, cam_ids):
        for track in tracker.tracked_stracks:
            if track.global_id < 0: continue
            result = {
                'cam_id': int(cam_id),
                'frame_id': frame_id,
                'track_id': track.global_id,
                'sct_track_id': track.track_id,
                'tlwh': list(map(lambda x: int(x), track.tlwh.tolist())),
                '2d_coord': track.location[0].tolist()
            }
            result_ = list(result.values())
            result_list.append(result)

def visualize(dets, img, colors, pose, pose_result, cur_frame):
    m = 2
    if len(dets) == 0:
        return img

    keypoints = [p['keypoints'][:,:2] for p in pose_result]
    scores = [p['keypoints'][:,2] for p in pose_result]
    img = visualize_kpt(img, keypoints, scores, thr=0.3)
            
    for obj in dets:
        score = obj[4]
        track_id = int(obj[5])
        len_feats = ' ' if obj[6] == 50 else obj[6]
        x0, y0, x1, y1 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

        color = (colors[track_id%80] * 255).astype(np.uint8).tolist()
        text = '{} : {:.1f}% | {}'.format(track_id, score * 100, len_feats)
        txt_color = (0, 0, 0) if np.mean(colors[track_id%80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4*m, 1*m)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (colors[track_id%80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - 1),
            (x0 + txt_size[0] + 1, y0 - int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4*m, txt_color, thickness=1*m)
    
    return img

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

    # draw keypoints and skeleton
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