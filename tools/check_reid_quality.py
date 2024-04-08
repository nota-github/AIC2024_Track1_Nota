from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random
import sys
import os
import numpy as np
sys.path.append('.')
from reids.fastreid.models.model import ReID
from tools.utils import sources
from trackers.multicam_tracker.matching import embedding_distance

def get_dists(same_id_dists, diff_id_dists, space_name, sets, reid, sample_num):
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
            if w < 50 or h < 150:
                continue
            b, r = t + w, l + h
            space = gt_txt.split("/")[4]
            cam = gt_txt.split("/")[5]
            img_path = f"/workspace/frames/{sets}/{space}/{cam}/{space}{cam}_{int(frame_id)}.jpg"
            id_info = [int(t), int(l), int(b), int(r), img_path]
            if obj_id in ids_infos:
                ids_infos[obj_id].append(id_info)
            else:
                ids_infos[obj_id] = [id_info]
    sorted_ids = sorted(ids_infos.keys())
    for i in sorted_ids:
        print(f"number of obj_id {i}: {len(ids_infos[i])}")

    # get cosine distances of same person
    for i in tqdm(range(sample_num)):
        person_id = random.sample(list(ids_infos.keys()), 1)[0]
        person_1, person_2 = random.sample(ids_infos[person_id], 2)
        img_1, img_2 = cv2.imread(person_1[4]), cv2.imread(person_2[4])

        g = 2.0
        img_1 = img_1.astype(np.float64)
        img_1 = ((img_1 / 255) ** (1 / g)) * 255  # gamma correction
        img_1 = img_1.astype(np.uint8)
        img_2 = img_2.astype(np.float64)
        img_2 = ((img_2 / 255) ** (1 / g)) * 255  # gamma correction
        img_2 = img_2.astype(np.uint8)

        bbox_1, bbox_2 = person_1[:4], person_2[:4]
        person1_feat = reid.run(img_1, [bbox_1], 1)
        person2_feat = reid.run(img_2, [bbox_2], 1)

        dist = embedding_distance(person1_feat, person2_feat) / 2.0  # adjust distance's range to 0 ~ 1
        same_id_dists.append(float(dist))

    # get cosine distances of different person
    for i in tqdm(range(sample_num)):
        person_id1, person_id2 = random.sample(list(ids_infos.keys()), 2)
        person_1, person_2 = random.sample(ids_infos[person_id1], 1)[0], random.sample(ids_infos[person_id2], 1)[0]
        img_1, img_2 = cv2.imread(person_1[4]), cv2.imread(person_2[4])
        bbox_1, bbox_2 = person_1[:4], person_2[:4]
        person1_feat = reid.run(img_1, [bbox_1], 1)
        person2_feat = reid.run(img_2, [bbox_2], 1)

        dist = embedding_distance(person1_feat, person2_feat) / 2.0  # adjust distance's range to 0 ~ 1
        diff_id_dists.append(float(dist))

def plot(same_list, diff_list, name):
    # Plot histograms for both lists
    plt.clf()
    plt.hist(same_list, alpha=0.5, bins=50, label='positive')
    plt.hist(diff_list, alpha=0.5, bins=50, label='negative')

    # Add legend and labels
    plt.legend()
    plt.xlabel('0.5 * Cosine Distance of Appearance Feature')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {name}')

    plt_path = f'tools/outputs/hist_{name}_reid.png'
    os.makedirs('tools/outputs', exist_ok=True)
    # Save plot
    plt.savefig(plt_path)
    print(f'histogram saved at {plt_path}')


if __name__ == "__main__":
    ReID.initialize(max_batch_size=1)
    reid = ReID()
    sample_num = 1000

    spaces = [
        "scene_041", "scene_042", "scene_043", "scene_044", "scene_045", "scene_046", "scene_047", "scene_048", "scene_049", "scene_050",
        "scene_051", "scene_052", "scene_053", "scene_054", "scene_055", "scene_056", "scene_057", "scene_058", "scene_059", "scene_060",
    ]
    # spaces = ["S005", "S008", "S013", "S017", "S020"]

    for space in spaces:
        same_id_dists = []
        diff_id_dists = []
        get_dists(same_id_dists, diff_id_dists, space, "val", reid, sample_num)
        plot(same_id_dists, diff_id_dists, space)
    
    # Use below when you want to draw a histogram of the entire place
    
    # same_id_dists = []
    # diff_id_dists = []
    # for space in spaces:
    #     get_dists(same_id_dists, diff_id_dists, space, "val", reid, sample_num)
    # plot(same_id_dists, diff_id_dists, "Total")
    
    ReID.finalize()
