import os
from tqdm import tqdm

dir_path = 'results'
output_file = 'track1_submission.txt'
output_path = os.path.join(dir_path, output_file)

result_paths = [
    'results/scene_061.txt',
    'results/scene_062.txt',
    'results/scene_063.txt',
    'results/scene_064.txt',
    'results/scene_065.txt',
    'results/scene_066.txt',
    'results/scene_067.txt',
    'results/scene_068.txt',
    'results/scene_069.txt',
    'results/scene_070.txt',
    'results/scene_071.txt',
    'results/scene_072.txt',
    'results/scene_073.txt',
    'results/scene_074.txt',
    'results/scene_075.txt',
    'results/scene_076.txt',
    'results/scene_077.txt',
    'results/scene_078.txt',
    'results/scene_079.txt',
    'results/scene_080.txt',
    'results/scene_081.txt',
    'results/scene_082.txt',
    'results/scene_083.txt',
    'results/scene_084.txt',
    'results/scene_085.txt',
    'results/scene_086.txt',
    'results/scene_087.txt',
    'results/scene_088.txt',
    'results/scene_089.txt',
    'results/scene_090.txt',
    ]

global_id_dict = {}
max_global_id = 1

with open(output_path, "w") as outfile:
    for result_path in tqdm(result_paths):
        with open(result_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            outfile.write(line)