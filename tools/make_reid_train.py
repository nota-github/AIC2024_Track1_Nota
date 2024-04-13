import os
import random
import cv2
from pathlib import Path
from tqdm import tqdm


train_gt_files = Path('data/videos/train').glob('**/ground_truth.txt')
last_obj_id = 0
os.makedirs(f'data/reid_dataset/bounding_box_train', exist_ok=True)

for gt_file in train_gt_files:
    print(gt_file)
    scene = str(gt_file).split('/')[2]
    with open(gt_file, 'r') as f:
        labels = f.readlines()
    obj_id_2_set = {}

    for label in tqdm(labels):
        cam_id, obj_id, frame_id, x, y, w, h, _, _ = label.split(' ')
        obj_id = int(obj_id) + last_obj_id + 1

        formatted_cam_id = f"{int(cam_id):04d}"
        formatted_obj_id = f"{int(obj_id):04d}"
        img_path = f"data/frames/train/{scene}/camera_{formatted_cam_id}/{scene}camera_{formatted_cam_id}_{frame_id}.jpg"
        x1, y1, x2, y2 = int(x), int(y), int(x) + int(w), int(y) + int(h)
        reid_img_name = f"{formatted_obj_id}_c{cam_id}s1_{frame_id}_00.jpg"
        
        to_add = [img_path, [x1,y1,x2,y2], reid_img_name]
        if obj_id in obj_id_2_set:
            obj_id_2_set[obj_id].append(to_add)
        else:
            obj_id_2_set[obj_id] = [to_add]

    for obj_id in tqdm(obj_id_2_set):
        sampled_objects = random.sample(obj_id_2_set[obj_id], 100)
        for obj in sampled_objects:
            im_path, box, reid_im_name = obj
            x1, y1, x2, y2 = box
            reid_path = f"data/reid_dataset/bounding_box_train/{reid_im_name}"
            img = cv2.imread(im_path)
            cropped_img = img[y1:y2, x1:x2]
            cv2.imwrite(reid_path, cropped_img)
        
    last_obj_id = max(obj_id_2_set.keys())