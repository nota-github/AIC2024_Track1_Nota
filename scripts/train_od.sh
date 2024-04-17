#!/bin/bash

export MKL_SERVICE_FORCE_INTEL=1
yolo train data=mcpt_detection.yaml model=yolov8x.pt epochs=30 imgsz=640 batch=16 device=0,1,2,3 name=aic24 exist_ok=True

cp runs/detect/aic24/weights/best.pt pretrained/yolov8x_aic.pt