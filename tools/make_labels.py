import os
from pathlib import Path
from tqdm import tqdm


def main(split):
    label_files = Path(f'data/videos/{split}').glob('**/ground_truth.txt')
    for l in label_files:
        label_folder_dir = str(l.parent)
        print(l, label_folder_dir)
        with open(l,'r') as f:
            labels = f.readlines()

        cam_sets = {}
        for label in tqdm(labels):
            cam_id, obj_id, frame_id, xmin, ymin, w, h, x_world, y_world = label.split(' ')
            label = ','.join([frame_id, obj_id, xmin, ymin, w, h, '1', x_world, y_world.split('\n')[0], '-1\n'])

            if cam_id in cam_sets:
                cam_sets[cam_id].append(label)
            else:
                cam_sets[cam_id] = [label]

        for cam_id in cam_sets:
            formatted_id = f"{int(cam_id):04d}"
            file_name = os.path.join(label_folder_dir, f'camera_{formatted_id}', f'label.txt')
            with open(file_name, 'w') as f:
                for t in cam_sets[cam_id]:
                    f.write(t)


if __name__ == "__main__":
    main('train')
    main('val')