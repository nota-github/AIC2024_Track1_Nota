# [CVPRW 2024] Cluster Self-Refinement for Enhanced Online Multi-Camera People Tracking

The official resitory for 8th NVIDIA AI City Challenge (Track1: Multi-Camera People Tracking) from team NetsPresso (Nota Inc.).

![mcpt_fig1](https://github.com/nota-github/AIC2024_Track1_Nota/assets/48690450/05d35d49-a218-452b-8db8-b5524d44a63f)


## Installation
```
# git clone this repository
git clone https://github.com/nota-github/AIC2024_Track1_Nota.git
cd AIC2024_Track1_Nota
```

## Prepare Datasets

The official dataset is available for download at https://www.aicitychallenge.org/2024-data-and-evaluation/, the website of the AI City Challenge.  
To get the password to download them, you must complete the dataset request form.  
(We are not permitted to share the dataset, per the DATASET LICENSE AGREEMENT from the dataset author(s).)

1. After unzipping the dataset zip files, make sure the data structure is as follows:

```
AIC2024_Track1_Nota
└── data
    └── videos
        ├── train
        │   ├── scene_001
        │   │   ├── camera_0001
        │   │   │   ├── calibration.json
        │   │   │   └── video.mp4
        │   │   ├── ...
        │   │   └── ground_truth.txt
        │   ├── scene_002
        │   ├── ...
        ├── val
        │   ├── ...
        └── test
            ├── ...
```

2. Generate datasets from videos
- Option1: In case you want to train object detection and re-identification models
```bash 
bash scripts/generate_all_datasets.sh
```

- Option2: In case you want to use pre-trained models
```bash 
bash scripts/generate_only_frames.sh
```

## Setup Environment
```bash
# Build a docker image
docker build -t aic2024/track1_nota:latest .

# Build a docker container
docker run -it --gpus all --shm-size=8g \
-v /path/to/AIC2024_Track1_Nota:/home/workspace/AIC2024_Track1_Nota \
-v /path/to/AIC2024_Track1_Nota/data:/workspace/ \
aic2024/track1_nota:latest /bin/bash

```


## Training
1. Train People Detection Model
- Modify the 'batch' and 'device' arguments in 'train_od.sh' based on the available GPUs.
```bash 
bash scripts/train_od.sh
```

2. Train ReID Model
- Modify the 'CUDA_VISIBLE_DEVICES' and 'num-gpus' arguments in 'train_reid.sh' based on the available GPUs.
- Download Market1501 pretrained weight from [here](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth) and place it in the './reid' directory.
```bash 
bash scripts/train_reid.sh
```

If you want to use pretrained models, please download them from the provided [Google Drive](https://drive.google.com/drive/folders/1f9vTZA336qr9JL8nPbA9fmH5hVU8rcuJ?usp=sharing) and place them in the './pretrained' directory. 


## Reproduce MCPT Results 
- Option1: Inference each scene sequentially
```bash 
bash scripts/run_mcpt.sh
```

- Option2: Inference scenes in parallel (to get a faster results)
    - modify the 'run_mcpt_parallel.sh' based on the number of available GPUs and the quantity of scenes you wish to process simultaneously.
```bash 
bash scripts/run_mcpt_parallel.sh
```
(If errors occur, inference only on the affected scenes separately, then run 'python3 tools/merge_results.py')  


The result files will be saved as follows:

```
AIC2024_Track1_Nota
└── results
    ├── scene_061.txt
    ├── ...
    └── track1_submission.txt
```
