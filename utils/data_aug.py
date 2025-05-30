import os
import argparse
import numpy as np
import math

def augment_cut_and_splice(X_preictal: np.ndarray, n_augmented: int) -> np.ndarray:
    """
    使用剪切拼接生成增强的 pre-ictal 样本，保留原始时间长度 T=1280。

    参数:
        X_preictal: ndarray, shape = (N, C, T)，原始 pre-ictal 样本
        n_augmented: int，生成的新样本数量

    返回:
        ndarray, shape = (n_augmented, C, T)，增强后的 pre-ictal 样本
    """
    N, C, T = X_preictal.shape
    num_parts = 3
    T_part = math.ceil(T / num_parts)  # 向上取整，保证拼接后至少等于 T，即1281
    augmented = []

    for _ in range(n_augmented):
        parts = []
        for _ in range(num_parts):
            # 原始数据格式（N, C, T）
            # 随机选取一条 EEG 样本（shape: (C, 1280)）。
            sample = X_preictal[np.random.randint(N)]
            # 在每个样本中随机选取 T_part 长度的片段,shape[1] = 1280, T_part = 427
            # start = 1280 - 427 + 1 = 854, 从 0 到 854 随机选取
            start = np.random.randint(0, sample.shape[1] - T_part + 1)
            parts.append(sample[:, start:start + T_part])
        new_sample = np.concatenate(parts, axis=1)[:, :T]  # 拼接后截断到 T=1280
        augmented.append(new_sample)

    return np.array(augmented, dtype=np.float32)

def process_and_save_augmented_data(input_file: str, output_path: str):
    """
    执行数据增强并保存为新的 .npz 文件，调整 pre:inter 比例为 3:2。
    """
    os.makedirs(output_path, exist_ok=True)
    data = np.load(input_file)
    X, y = data["X"], data["y"]

    X_pre = X[y == 1]
    X_inter = X[y == 0]

    n_pre = len(X_pre)
    n_inter_target = 2 * n_pre
    n_pre_target = 3 * n_pre
    n_augmented = n_pre_target - n_pre

    # 下采样 inter-ictal
    if len(X_inter) >= n_inter_target:
        idx = np.random.choice(len(X_inter), n_inter_target, replace=False)
        X_inter_sampled = X_inter[idx]
    else:
        raise ValueError(f"inter-ictal 样本不足（当前 {len(X_inter)}），无法满足 2:1 下采样。")

    y_inter_sampled = np.zeros(n_inter_target, dtype=np.int64)

    # 生成增强的 pre-ictal
    X_aug = augment_cut_and_splice(X_pre, n_augmented)
    y_aug = np.ones(n_augmented, dtype=np.int64)

    # 合并最终数据
    X_final = np.concatenate([X_pre, X_aug, X_inter_sampled], axis=0)
    y_final = np.concatenate([np.ones(n_pre), y_aug, y_inter_sampled], axis=0)

    # 保存
    save_path = os.path.join(output_path, f"_segments_augmented.npz")
    np.savez_compressed(save_path, X=X_final, y=y_final)
    print(f"✅ 增强数据已保存至 {save_path}，最终比例 pre:inter = {len(X_final[y_final==1])}:{len(X_final[y_final==0])}")


def main(args):
    base_input = args.input_dir  # 保留初始输入路径
    base_output = args.output_dir  # 保留初始输出路径

    for pid_num in range(1, 24):
        patient_id = f"chb{pid_num:02d}"
        input_file = os.path.join(base_input, f"{patient_id}_segments.npz")
        output_dir = os.path.join(base_output, patient_id)

        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 正在处理患者 {patient_id} 的数据...")

        process_and_save_augmented_data(
            input_file=input_file,
            output_path=output_dir
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance pre-ictal EEG samples to generate a balanced training dataset.")
    parser.add_argument('--input_dir', type=str, default='data/processed', help='Directory containing input .npz files')
    parser.add_argument('--output_dir', type=str, default='data/augmented', help='Directory to save augmented data')

    args = parser.parse_args()
    main(args)
    
