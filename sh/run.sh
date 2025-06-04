nohup python utils/CSP.py
    > logs/feature_0531.log 2>&1 &


nohup python train_kf.py \
    > logs/trainkf_0603_morelr.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python train_lo.py \
    > logs/trainlo_0601.log 2>&1 &


nohup python train_kf.py \
    > logs/trainmlp_0603.log 2>&1 &

    
nohup python -m train.train_lr     \
    > logs/train_lr_0603.log 2>&1 &


nohup python -m train.train_kf     \
    > logs/train_oriCNN_betterKalman.log 2>&1 &


nohup python -m train.train_svm     \
    > logs/train_svm_betterKalman.log 2>&1 &


nohup python -m train.train_lr     \
    > logs/train_lr_betterKalman.log 2>&1 &

nohup python -m train.train_mlp     \
    > logs/train_mlp_betterKalman.log 2>&1 &


