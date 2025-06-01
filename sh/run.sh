nohup python utils/CSP.py
    > logs/feature_0531.log 2>&1 &


nohup python train_kf.py \
    > logs/trainkf_0602_morelr.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python train_lo.py \
    > logs/trainlo_0601.log 2>&1 &
