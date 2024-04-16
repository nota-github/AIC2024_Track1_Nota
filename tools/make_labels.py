import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_label_file(label_file):
    label_folder_dir = str(label_file.parent)
    print(label_file, label_folder_dir)
    with open(label_file, 'r') as f:
        labels = f.readlines()

    cam_sets = {}
    for label in labels:
        cam_id, obj_id, frame_id, xmin, ymin, w, h, x_world, y_world = label.split()
        label = ','.join([frame_id, obj_id, xmin, ymin, w, h, '1', x_world, y_world.strip(), '-1\n'])

        if cam_id in cam_sets:
            cam_sets[cam_id].append(label)
        else:
            cam_sets[cam_id] = [label]

    for cam_id in cam_sets:
        formatted_id = f"{int(cam_id):04d}"
        os.makedirs(os.path.join(label_folder_dir, f'camera_{formatted_id}'), exist_ok=True)
        file_name = os.path.join(label_folder_dir, f'camera_{formatted_id}', 'label.txt')
        with open(file_name, 'w') as f:
            f.writelines(cam_sets[cam_id])

def main(split, num_processes=4):
    label_files = list(Path(f'data/videos/{split}').glob('**/ground_truth.txt'))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        list(tqdm(executor.map(process_label_file, label_files), total=len(label_files)))

if __name__ == "__main__":
    main('train')
    main('val')
