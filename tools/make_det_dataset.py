import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_labels(label_file, split):
    label_folder_dir = str(label_file.parent)
    frames_dir = label_folder_dir.replace('videos', 'frames')
    frames = sorted(Path(frames_dir).glob('**/*.jpg'), key=lambda x: int(x.stem.split('_')[-1]))
    
    txt_contents = []
    prev_frame_id = 0
    
    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        frame_id, obj_id, xmin, ymin, w, h, *rest = map(float, label.split(','))
        frame_id = int(frame_id)
        
        # Constants for converting bounding box coordinates
        height, width = 1080, 1920
        dw, dh = 1.0 / width, 1.0 / height
        x, y = (xmin + w / 2.0) * dw, (ymin + h / 2.0) * dh
        w, h = w * dw, h * dh

        line = f"{0} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n"
        
        if prev_frame_id != frame_id and prev_frame_id != 0:
            if prev_frame_id % 30 == 0:
                file_name = f"data/od_total/{split}/labels/{Path(frames[prev_frame_id-1]).name.replace('.jpg', '.txt')}"
                with open(file_name, 'w') as f:
                    f.writelines(txt_contents)
                shutil.copy(frames[prev_frame_id-1], file_name.replace('labels', 'images').replace('.txt', '.jpg'))
            txt_contents = []

        txt_contents.append(line)
        prev_frame_id = frame_id

def main(split, num_processes=4):
    label_files = list(Path(f'data/videos/{split}').glob('**/label.txt'))
    os.makedirs(f'data/od_total/{split}/images', exist_ok=True)
    os.makedirs(f'data/od_total/{split}/labels', exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        list(tqdm(executor.map(process_labels, label_files, [split]*len(label_files)), total=len(label_files)))

if __name__ == "__main__":
    main('train')
    main('val')
