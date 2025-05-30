import os
import argparse
import numpy as np
import pywt
from scipy.linalg import eigh

# ---------------------
# Wavelet packet decomposition
# ---------------------
def wavelet_packet_decompose(signal, level=3, wavelet='db4'):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = wp.get_level(level, order='freq')
    return [node.data for node in nodes]

# ---------------------
# Covariance matrix of a trial
# ---------------------
def compute_covariance(trial):
    trial = trial - np.mean(trial, axis=1, keepdims=True)
    cov = trial @ trial.T
    return cov / np.trace(cov)

# ---------------------
# CSP: compute projection matrix
# ---------------------
def csp(X1, X2, m=2):
    # 计算每个样本的协方差矩阵；对所有样本求平均，得到两个类别的平均协方差矩阵 C1, C2；
    C1 = np.mean([compute_covariance(x) for x in X1], axis=0)
    C2 = np.mean([compute_covariance(x) for x in X2], axis=0)
    Cc = C1 + C2
    # 对 Cc 做特征分解（对称矩阵 → eigh）； 构造白化矩阵 P：使得 P @ Cc @ P.T ≈ I（单位矩阵）；
    # 白化的目的是消除不同通道间的线性相关性。
    eigvals, U = eigh(Cc)
    P = U @ np.diag(1.0 / np.sqrt(eigvals)) @ U.T
    S1 = P @ C1 @ P.T
    eigvals_s1, B = eigh(S1)
    W = B.T @ P
    W = np.concatenate((W[:m], W[-m:]), axis=0)
    return  W

# ---------------------
# Extract features for one EEG trial
# ---------------------
def extract_feature_matrix(trial, csp_filters_list):
    split = trial.shape[1] // 2
    segments = [trial[:, :split], trial[:, split:]]
    features = []

    for seg in segments:
        band_features = []
        bands = [seg] + wavelet_packet_decompose(seg, level=3)
        for band, W in zip(bands, csp_filters_list):
            projected = W @ band
            var = np.var(projected, axis=1)
            var_norm = np.log(var / np.sum(var))
            band_features.append(var_norm)
        features.append(np.concatenate(band_features))
    return np.vstack(features)  # shape: (2, 36)

# ---------------------
# Process one file
# ---------------------
def process_file(input_path, output_path):
    data = np.load(input_path)
    X, y = data['X'], data['y']
    X_pre = X[y == 1]
    X_inter = X[y == 0]

    print(f"📊 正在训练 CSP filters（pre={len(X_pre)}, inter={len(X_inter)})")
    csp_filters_list = []
    # 对原始数据进行小波包分解，提取 9 个频带
    for band_idx in range(9):
        def extract_band(X):
            return np.array([wavelet_packet_decompose(x, level=3)[band_idx-1] if band_idx > 0 else x for x in X])
        band_pre = extract_band(X_pre)
        band_inter = extract_band(X_inter)
        W = csp(band_pre, band_inter, m=2)
        csp_filters_list.append(W)

    print(f"🚀 开始特征提取，共 {len(X)} 条片段")
    F = np.array([extract_feature_matrix(x, csp_filters_list) for x in X], dtype=np.float32)

    os.makedirs(output_path, exist_ok=True)
    save_name = os.path.basename(input_path).replace('_segments_augmented.npz', 'features.npz')
    np.savez_compressed(os.path.join(output_path, save_name), F=F, y=y)
    print(f"✅ 已保存特征到 {save_name}，shape={F.shape}")

# ---------------------
# 主函数
# ---------------------
def main(args):
    input_root = args.input_dir
    output_root = args.output_dir

    for patient_id in sorted(os.listdir(input_root)):
        input_file = os.path.join(input_root, patient_id, f"_segments_augmented.npz")
        if not os.path.exists(input_file):
            print(f"⚠️ 文件缺失：{input_file}")
            continue
        output_dir = os.path.join(output_root, patient_id)
        print(f"\n📂 处理 {patient_id} 中的增强数据...")
        process_file(input_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CSP-based EEG features from .npz dataset.")
    parser.add_argument('--input_dir', type=str, default='data/augmented', help='Directory with input .npz files')
    parser.add_argument('--output_dir', type=str, default='data/features', help='Directory to save feature npz files')
    args = parser.parse_args()
    main(args)
