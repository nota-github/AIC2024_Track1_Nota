#!/bin/bash

# extract frames from videos
python3 tools/extract_frames.py --path data/videos --num-processes 4
# split ground_truth.txt
python3 tools/make_labels.py
# make people detection dataset
python3 tools/make_det_dataset.py
# make re-identification datasets
python3 tools/make_reid_train.py
python3 tools/make_reid_val.py