import os
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
from datetime import timedelta
import json
import re
from collections import defaultdict
import argparse
from typing import List, Tuple, Dict, Generator



def butter_bandpass_filter(data, lowcut=5.0, highcut=50.0, fs=256.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def read_annotations(summary_path):
    """
    解析 seizure summary 文件
    返回 seizure 的记录名、开始时间、结束时间
    """
    seizures = []
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    current_record = None
    for line in lines:
        if "File Name" in line:
            current_record = line.split(":")[1].strip()
        elif "Seizure Start Time" in line:
            start = int(re.search(r"\d+", line).group())
        elif "Seizure End Time" in line:
            end = int(re.search(r"\d+", line).group())
            seizures.append((current_record, start, end))
    return seizures

def label_segments(record_name: str,
                   total_duration: float,
                   seizure_times: List[Tuple[str, int, int]],
                   edf_durations: Dict[str, float],
                   window_size: int = 5 * 256,
                   pre_ictal_window: int = 1800) -> Generator:
    """
    生成 EEG 标签，必要时从前一个记录中补足 pre-ictal 区间。
    """
    num_windows = int(total_duration // (window_size / 256))
    labels = np.zeros(num_windows, dtype=int)

    for _, seizure_start, _ in seizure_times:
        if seizure_start >= pre_ictal_window:
            # 本文件足够提供完整的 pre-ictal 区间
            start_win = int((seizure_start - pre_ictal_window) // (window_size / 256))
            end_win = int(seizure_start // (window_size / 256))
            labels[start_win:end_win] = 1
        else:
            # 只标注当前文件能提供的部分
            end_win = int(seizure_start // (window_size / 256))
            labels[0:end_win] = 1
            # 尝试从上一个文件补
            record_list = list(edf_durations.keys())
            idx = record_list.index(record_name)
            if idx > 0:
                prev_name = record_list[idx - 1]
                prev_duration = edf_durations[prev_name]
                missing = pre_ictal_window - seizure_start
                prev_needed_windows = int(missing // (window_size / 256))
                prev_num_windows = int(prev_duration // (window_size / 256))
                prev_start_win = max(prev_num_windows - prev_needed_windows, 0)
                yield (prev_name, prev_start_win, prev_num_windows)
    yield (record_name, labels)


def process_edf(file_path: str,
                seizure_times: List[Tuple[str, int, int]],
                fs: int,
                window_size: int,
                target_channels: List[str],
                label_override: np.ndarray = None) -> List[Tuple[np.ndarray, int]]:
    """
    提取 EDF 数据并返回切片及标签。
    """
    f = pyedflib.EdfReader(file_path)
    # 返回该 EDF 文件中所有通道的名称（标签）
    signal_labels = f.getSignalLabels()
    # 去重通道选择逻辑：按 target_channels 顺序，选取 signal_labels 中首次出现的索引
    channel_indices = []
    seen = set()
    for ch in target_channels:
        if ch in signal_labels and ch not in seen:
            idx = signal_labels.index(ch)
            channel_indices.append(idx)
            seen.add(ch)

    total_samples = f.getNSamples()[0]
    total_duration = total_samples / fs

    # 获取标签
    if label_override is not None:
        labels = label_override
    else:
        raise ValueError("label_override 必须提供，否则无法生成标签。")

    # 读取并滤波信号
    filtered_signals = []
    for idx in channel_indices:
        raw = f.readSignal(idx)
        filtered = butter_bandpass_filter(raw, 5, 50, fs)
        filtered_signals.append(filtered)
    f._close()

    data = np.array(filtered_signals)  # shape: (channels, time)
    segments = []

    num_windows = len(labels)
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        if end <= data.shape[1]:
            segment = data[:, start:end]
            segments.append((segment, labels[i]))
    return segments


def save_patient_data(patient_id: str, segments: List[Tuple[np.ndarray, int]], save_dir: str = "data/processed"):
    """
    保存某个患者的所有 EEG 段及对应标签为 .npz 文件。
    
    参数:
        patient_id: 患者 ID，如 'chb01'
        segments: [(EEG 段, 标签)] 的列表，每段为 (channels, samples),chb01的所有 EEG 段
        save_dir: 保存目录，默认是 data/processed
    """
    os.makedirs(save_dir, exist_ok=True)

    X = np.array([seg for seg, _ in segments], dtype=np.float32)  # shape: (N, C, T)
    y = np.array([label for _, label in segments], dtype=np.int64)  # shape: (N,)

    save_path = os.path.join(save_dir, f"{patient_id}_segments.npz")
    np.savez_compressed(save_path, X=X, y=y)

    print(f"✅ 数据已保存至 {save_path}，共 {X.shape[0]} 个样本，shape: {X.shape}")

def main(args):
    fs = args.fs
    window_size = args.window_size * fs
    pre_ictal_window = args.pre_ictal_window * 60

    patient_path = os.path.join(args.root, args.patient_id)
    summary_path = args.summary_path or os.path.join(patient_path, f"{args.patient_id}-summary.txt")
    # 返回 seizure 的记录名、开始时间、结束时间,注意时间都是以秒为单位
    seizure_info = read_annotations(summary_path)

    edf_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.edf')])
    edf_paths = [os.path.join(patient_path, f) for f in edf_files]

    edf_durations = {}
    for path in edf_paths:
        f = pyedflib.EdfReader(path)
        duration = f.getNSamples()[0] / f.getSampleFrequency(0)
        edf_durations[os.path.basename(path)] = duration
        f._close()

    label_map = {}

    for file in edf_files:
        # 之前记录了哪些file是有癫痫发作的，获取当前 file 的癫痫发作信息，如果没有则为空列表
        seizures = [(name, s, e) for (name, s, e) in seizure_info if name == file]
        # 对于 file 中的每个记录，生成标签
        for result in label_segments(file, edf_durations[file], seizures, edf_durations, window_size, pre_ictal_window):
            if isinstance(result[1], np.ndarray):
                label_map[result[0]] = result[1]
            else:
                fname, start_win, end_win = result
                total_win = int(edf_durations[fname] // (window_size / fs))
                if fname not in label_map:
                    label_map[fname] = np.zeros(total_win, dtype=int)
                label_map[fname][start_win:end_win] = 1

    # 按标签提取片段
    all_segments = []
    for file in edf_files:
        full_path = os.path.join(patient_path, file)
        if file in label_map:
            segments = process_edf(
                full_path,
                seizure_times=[],
                fs=fs,
                window_size=window_size,
                target_channels=args.target_channels,
                label_override=label_map[file]
            )
            all_segments.extend(segments)

    print(f"✅ 总共获得 {len(all_segments)} 个片段（{args.patient_id}）")
    save_patient_data(args.patient_id, all_segments, save_dir=args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG data and generate segments with labels.")
    parser.add_argument('--root', type=str, default='data/raw/chbmit_dataset', help='Root directory of the dataset')
    parser.add_argument('--patient_id', type=str, default='chb01', help='Patient ID to process')
    parser.add_argument('--fs', type=int, default=256, help='Sampling frequency in Hz')
    parser.add_argument('--window_size', type=int, default=5, help='Window size in seconds')
    parser.add_argument('--pre_ictal_window', type=int, default=30, help='Pre-ictal window size in minutes')
    parser.add_argument('--target_channels', type=str, nargs='+', default=[
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ], help='List of target channels to process')
    parser.add_argument('--summary_path', type=str, default='', help='Path to the seizure summary file (optional)')
    parser.add_argument('--save_dir', type=str, default='data/processed', help='Directory to save processed data')

     # 解析参数

    args = parser.parse_args()
    main(args)
