import os
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil


def main(split):
    label_files = Path(f'data/videos/{split}').glob('**/label.txt')
    label_files = Path(f'data/videos/{split}').glob('**/label.txt')
    os.makedirs(f'data/od_total/{split}/images', exist_ok=True)
    os.makedirs(f'data/od_total/{split}/labels', exist_ok=True)

    for l in tqdm(label_files):
        label_folder_dir = str(l.parent)
        with open(l,'r') as f:
            labels = f.readlines()
        frames_dir = label_folder_dir.replace('videos', 'frames')
        frames = Path(frames_dir).glob('**/*.jpg')
        frames = [str(f) for f in frames]
        frames = sorted(frames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        txt_contents = []
        prev_frame_id = 0
        for label in labels:
            frame_id, obj_id, xmin, ymin, w, h, _, _, _, _ = label.split(',')
            frame_id, obj_id, xmin, ymin, w, h = int(frame_id), int(obj_id), float(xmin), float(ymin), float(w), float(h)

            height, width, _ = 1080, 1920, 3
            dw = 1. / width
            dh = 1. / height
            x, y = xmin + w / 2.0, ymin + h / 2.0
            x, y, w, h = x*dw, y*dh, w*dw, h*dh
            line = f"{0} {round(x, 5)} {round(y, 5)} {round(w, 5)} {round(h, 5)}\n"
            if prev_frame_id != frame_id and prev_frame_id != 0:
                if prev_frame_id % 30 == 0:  # train: 30 / val: 30
                    file_name = f"data/od_total/{split}/labels/" + frames[prev_frame_id-1].split('/')[-1].replace('.jpg','.txt')
                    with open(file_name, 'w') as f:
                        for t in txt_contents:
                            f.write(t)
                    txt_contents = []
                    shutil.copy(frames[prev_frame_id-1], file_name.replace('labels','images').replace('.txt','.jpg'))
                else:
                    txt_contents = []

            txt_contents.append(line)
            prev_frame_id = frame_id


if __name__ == "__main__":
    main('train')
    main('val')
        
        