# extract frames from videos
python3 tools/extract_frames.py --path data/videos
# split ground_truth.txt
python3 make_labels.py
# make people detection dataset
python3 make_det_dataset.py
# make re-identification datasets
python3 make_reid_train.py
python3 make_reid_val.py