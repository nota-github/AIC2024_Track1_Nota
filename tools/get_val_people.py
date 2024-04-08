from pathlib import Path
from tqdm import tqdm
import cv2
import random
import sys
import os
sys.path.append('.')
from tools.utils import sources

def crop_people(space_name, sets):
    print(f"\nStart {sets}/{space_name}")

    gt_txts = Path(f"/workspace/videos/{sets}/{space_name}").glob("**/*.txt")
    gt_txts = [str(p) for p in gt_txts if 'label' in str(p)]

    ids_infos = {}
    for gt_txt in gt_txts:
        # print('@', gt_txt)
        with open(gt_txt, 'r') as f:
            labels = f.readlines()
        for l in labels:
            frame_id, obj_id, t, l, w, h, _, _, _, _ = l.split(",")
            frame_id, obj_id = int(frame_id), int(obj_id)
            t, l, w, h = float(t), float(l), float(w), float(h)
            # Filtering abnormal boxes
            # if w < 50 or h < 150:
            #     continue
            b, r = t + w, l + h
            space = gt_txt.split("/")[4]
            cam = gt_txt.split("/")[5]
            img_path = f"/workspace/frames/{sets}/{space}/{cam}/{space}{cam}_{int(frame_id)}.jpg"
            id_info = [int(t), int(l), int(b), int(r), img_path, frame_id]
            if obj_id in ids_infos:
                ids_infos[obj_id].append(id_info)
            else:
                ids_infos[obj_id] = [id_info]
 
    for i, obj_id in enumerate(ids_infos):
        # max_area = 0
        # crop_obj = None
        # min_frame_id = 24000
        # for obj in ids_infos[obj_id]:
        #     t, l, b, r, _, frame_id = obj
        #     if frame_id < min_frame_id:
        #         min_frame_id = frame_id
        #         crop_obj = obj

        sorted_frameid = sorted(ids_infos[obj_id], key=lambda x: x[5])
        sorted_area = sorted(ids_infos[obj_id], key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

            # area = (b-t) * (r-l)
            # if area > max_area:
            #     max_area = area
            #     crop_obj = obj
        t, l, b, r, img_path, frame_id = sorted_frameid[0]
        img = cv2.imread(img_path)
        crop = img[l:r, t:b]
        dst_path = f'tools/outputs/cluster_people/{space_name}'
        os.makedirs(dst_path, exist_ok=True)
        cv2.imwrite(f'{dst_path}/{i+1}_first_{frame_id}.jpg', crop)

        t, l, b, r, img_path, frame_id = sorted_frameid[-1]
        img = cv2.imread(img_path)
        crop = img[l:r, t:b]
        dst_path = f'tools/outputs/cluster_people/{space_name}'
        os.makedirs(dst_path, exist_ok=True)
        cv2.imwrite(f'{dst_path}/{i+1}_last_{frame_id}.jpg', crop)

        t, l, b, r, img_path, frame_id = sorted_area[0]
        img = cv2.imread(img_path)
        crop = img[l:r, t:b]
        dst_path = f'tools/outputs/cluster_people/{space_name}'
        cv2.imwrite(f'{dst_path}/{i+1}_max_{frame_id}.jpg', crop)


if __name__ == "__main__":

    spaces = [
        "scene_041", "scene_042", "scene_043", "scene_044", "scene_045", "scene_046", "scene_047", "scene_048", "scene_049", "scene_050",
        "scene_051", "scene_052", "scene_053", "scene_054", "scene_055", "scene_056", "scene_057", "scene_058", "scene_059", "scene_060",
    ]
    
    for space in spaces:
        crop_people(space, "val")
    