from pathlib import Path
import cv2
import sys
import os
sys.path.append('.')
from reids.fastreid.models.model import ReID
from detection.models.model import NPNet
from tools.utils import sources, cam_ids, get_reader_writer


def crop_people(det, scene, sources, cam_ids):
    src_handlers = [get_reader_writer(s) for s in sources]

    for (img_paths, writer), cam_id in zip(src_handlers, cam_ids):
        img_path = img_paths.pop(0)
        img = cv2.imread(img_path)
        dst_folder = f'tools/outputs/crop_people/{scene}'
        os.makedirs(dst_folder, exist_ok=True)

        dets = detection.run(img)

        for i, det in enumerate(dets):
            x1, y1, x2, y2, _, _ = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(dst_folder, f'cam{cam_id}_{i}.jpg'), cropped_img)
    
    return


if __name__ == "__main__":
    NPNet.initialize()
    ReID.initialize(max_batch_size=1)
    
    detection = NPNet()
    detection.conf_thres = 0.10
    detection.iou_thres = 0.45
    reid = ReID()

    scenes = [
        "scene_041", "scene_042", "scene_043", "scene_044", "scene_045", "scene_046", "scene_047", "scene_048", "scene_049", "scene_050",
        "scene_051", "scene_052", "scene_053", "scene_054", "scene_055", "scene_056", "scene_057", "scene_058", "scene_059", "scene_060",
    ]

    for scene in scenes:
        crop_people(detection, scene, sources[scene], cam_ids[scene])
        break
    
    NPNet.finalize()
    ReID.finalize()