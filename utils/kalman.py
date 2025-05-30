from pykalman import KalmanFilter

def kalman_smooth(pred_probs):
    """
    对预测概率进行卡尔曼滤波，输入为 (N,) 的 pre-ictal 概率序列。
    返回平滑后的序列。
    """
    kf = KalmanFilter(initial_state_mean=0.5, n_dim_obs=1)
    smoothed_state_means, _ = kf.smooth(pred_probs.reshape(-1, 1))
    return smoothed_state_means.ravel()

# 示例：模型预测输出 -> 平滑
model.eval()
with torch.no_grad():
    preds = model(X_tensor)
    pre_probs = preds[:, 1].numpy()
    pre_probs_smooth = kalman_smooth(pre_probs)

# 阈值触发报警（可选）
threshold = 0.6
pred_final = (pre_probs_smooth > threshold).astype(int)
