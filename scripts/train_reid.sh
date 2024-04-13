cd reid
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file configs/AIC24/sbs_R50-ibn_ws.yml --num-gpus 2 MODEL.WEIGHTS market_sbs_R50-ibn.pth

cp logs/aic24/sbs_R50-ibn/model_best.pth ../pretrained/market_aic_sbs_R50-ibn.pth