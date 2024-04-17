#!/bin/bash

SCRIPT="python3 eval_mcpt.py"

CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_061 > logs/log_s61.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_062 > logs/log_s62.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $SCRIPT --scene scene_063 > logs/log_s63.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $SCRIPT --scene scene_064 > logs/log_s64.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 $SCRIPT --scene scene_065 > logs/log_s65.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $SCRIPT --scene scene_066 > logs/log_s66.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $SCRIPT --scene scene_067 > logs/log_s67.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $SCRIPT --scene scene_068 > logs/log_s68.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_069 > logs/log_s69.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_070 > logs/log_s70.txt 2>&1

# Wait for all to finish
wait
echo "Scene061-070 processes have completed."

CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_071 > logs/log_s71.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_072 > logs/log_s72.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $SCRIPT --scene scene_073 > logs/log_s73.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $SCRIPT --scene scene_074 > logs/log_s74.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 $SCRIPT --scene scene_075 > logs/log_s75.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $SCRIPT --scene scene_076 > logs/log_s76.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $SCRIPT --scene scene_077 > logs/log_s77.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $SCRIPT --scene scene_078 > logs/log_s78.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_079 > logs/log_s79.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_080 > logs/log_s80.txt 2>&1

# Wait for all to finish
wait
echo "Scene071-080 processes have completed."

CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_081 > logs/log_s81.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_082 > logs/log_s82.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $SCRIPT --scene scene_083 > logs/log_s83.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $SCRIPT --scene scene_084 > logs/log_s84.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 $SCRIPT --scene scene_085 > logs/log_s85.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $SCRIPT --scene scene_086 > logs/log_s86.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $SCRIPT --scene scene_087 > logs/log_s87.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $SCRIPT --scene scene_088 > logs/log_s88.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_089 > logs/log_s89.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_090 > logs/log_s90.txt 2>&1

# Wait for all to finish
wait
echo "All processes have completed."

# merge results for submission
python3 tools/merge_results.py