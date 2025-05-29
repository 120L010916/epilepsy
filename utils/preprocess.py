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


# 五阶巴特沃斯带通滤波器
def butter_bandpass_filter(data, lowcut=5.0, highcut=50.0, fs=256.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def read_annotations(summary_path: str) -> Tuple[List[Tuple[str, int, int]], Dict[str, Tuple[str, str]]]:
    """
    解析 seizure summary 文件，提取：
    1. seizures: List of (record_name, seizure_start_seconds, seizure_end_seconds)
    2. file_time_ranges: Dict of record_name -> (start_time_str, end_time_str)
    """
    seizures = []
    file_time_ranges = {}

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_record = None
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_record = line.split(":", 1)[1].strip()
        elif line.startswith("File Start Time:"):
            start_time = line.split(":", 1)[1].strip()
        elif line.startswith("File End Time:"):
            end_time = line.split(":", 1)[1].strip()
            # 在读取完结束时间之后就可以记录当前文件的时间范围
            if current_record:
                file_time_ranges[current_record] = (start_time, end_time)
        elif line.startswith("Seizure Start Time:"):
            seizure_start = int(re.search(r"\d+", line).group())
        elif line.startswith("Seizure End Time:"):
            seizure_end = int(re.search(r"\d+", line).group())
            if current_record:
                seizures.append((current_record, seizure_start, seizure_end))

    return seizures, file_time_ranges

from typing import Generator, List, Tuple, Dict
import numpy as np
from datetime import datetime

def label_segments(record_name: str,
                   total_duration: float,
                   seizure_times: List[Tuple[str, int, int]],
                   edf_durations: Dict[str, float],
                   edf_start_times: Dict[str, datetime],
                   window_size: int = 5 * 256,
                   pre_ictal_window: int = 1800) -> Generator:
    """
    生成 EEG 标签，必要时从前一个记录中补足 pre-ictal 区间。
    若前一个记录与当前记录之间时间间隔过大，则不进行补足。
    """
    num_windows = int(total_duration // (window_size / 256))
    labels = np.zeros(num_windows, dtype=int)
    broken_window = pre_ictal_window / 2
    for _, seizure_start, _ in seizure_times:
        if seizure_start >= pre_ictal_window:
            # 当前记录可以完整提供 pre-ictal 区间
            start_win = int((seizure_start - pre_ictal_window) // (window_size / 256))
            end_win = int(seizure_start // (window_size / 256))
            labels[start_win:end_win] = 1
        else:
            # 当前记录部分标注
            end_win = int(seizure_start // (window_size / 256))
            labels[0:end_win] = 1

            # 检查是否需要补足，且是否时间允许
            record_list = list(edf_durations.keys())
            idx = record_list.index(record_name)

            if idx > 0:
                prev_name = record_list[idx - 1]
                prev_duration = edf_durations[prev_name]

                # 检查时间断裂
                current_start = edf_start_times[record_name]
                prev_start = edf_start_times[prev_name]
                prev_end = prev_start + timedelta(seconds=prev_duration)
                time_gap = (current_start - prev_end).total_seconds()

                if time_gap <= broken_window:
                    # 前一个记录时间接近，可以补充
                    # missing: 当前记录中还差多少秒用于补足 pre-ictal。
                    missing = pre_ictal_window - seizure_start
                    prev_needed_windows = int(missing // (window_size / 256))
                    prev_num_windows = int(prev_duration // (window_size / 256))
                    prev_start_win = max(prev_num_windows - prev_needed_windows, 0)
                    yield (prev_name, prev_start_win, prev_num_windows)
                else:
                    # 时间断裂过大，无法补充
                    pass  # 可以打印日志或记录信息

    yield (record_name, labels)



def process_edf(file_path: str,
                seizure_times: List[Tuple[str, int, int]],
                fs: int,
                window_size: int,
                target_channels: List[str],
                label_override: np.ndarray = None) -> List[Tuple[np.ndarray, int]]:
    """
    提取 EDF 数据并返回切片及标签。
    提取指定通道的数据
    对数据进行滤波处理
    按照固定时间窗口切片
    并 结合外部提供的标签（label_override）生成标注的数据片段
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
   
    # 获取标签
    if label_override is not None:
        labels = label_override
    else:
        raise ValueError("label_override 必须提供，否则无法生成标签。")

    # 读取并滤波信号
    filtered_signals = []
    for idx in channel_indices:
        raw = f.readSignal(idx)
        # 使用五阶巴特沃斯带通滤波器
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

    for pid_num in range(1, 24):  # chb01 到 chb23
        patient_id = f"chb{pid_num:02d}"
        patient_path = os.path.join(args.root, patient_id)
        summary_path = os.path.join(patient_path, f"{patient_id}-summary.txt")

        if not os.path.exists(summary_path):
            print(f"⚠️ 跳过 {patient_id}，未找到 summary 文件")
            continue

        print(f"\n🚀 正在处理 {patient_id}...")

        # 读取癫痫发作标注,返回记录名、开始时间、结束时间
        seizure_info, file_time_ranges = read_annotations(summary_path)
        edf_start_times = {}
        for record_name, (start_time_str, _) in file_time_ranges.items():
            # 假设日期统一使用默认日期（1900-01-01），仅用于时间差计算
            start_dt = datetime.strptime(start_time_str, "%H:%M:%S")
            edf_start_times[record_name] = start_dt
        # 获得每个edf文件的持续时间
        edf_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.edf')])
        edf_durations = {}

        for fname in edf_files:
            fpath = os.path.join(patient_path, fname)
            try:
                f = pyedflib.EdfReader(fpath)
                duration = f.getNSamples()[0] / f.getSampleFrequency(0)
                edf_durations[fname] = duration
                f._close()
            except Exception as e:
                print(f"⚠️ EDF 打开失败 {fname}：{e}")
                continue

        # 构造标签：在每个 EDF 文件中标注癫痫发作的时间-窗口，获得标签为1
        # 这里假设每个 EDF 文件的时间戳是连续的
        # 如果开始时间-窗口不够，需要从前一个文件补足 pre-ictal 区间
        label_map = {}
        for file in edf_files:
            seizures = [(name, s, e) for (name, s, e) in seizure_info if name == file]
            for result in label_segments(file, edf_durations[file], seizures, edf_durations, edf_start_times, window_size, pre_ictal_window):
                if isinstance(result[1], np.ndarray):
                    label_map[result[0]] = result[1]
                else:
                    fname, start_win, end_win = result
                    total_win = int(edf_durations[fname] // (window_size / fs))
                    if fname not in label_map:
                        label_map[fname] = np.zeros(total_win, dtype=int)
                    label_map[fname][start_win:end_win] = 1

        # 提取片段
        all_segments = []
        for file in edf_files:
            full_path = os.path.join(patient_path, file)
            if file in label_map:
                segments = process_edf(
                    file_path=full_path,
                    seizure_times=[],
                    fs=fs,
                    window_size=window_size,
                    target_channels=args.target_channels,
                    label_override=label_map[file]
                )
                all_segments.extend(segments)

        print(f"✅ {patient_id} 处理完成，获得 {len(all_segments)} 个片段")
        # 保存为.npz格式
        if all_segments:
            save_patient_data(patient_id, all_segments, save_dir=args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG data and generate segments with labels.")
    parser.add_argument('--root', type=str, default='data/raw/chbmit_dataset', help='Root directory of the dataset')
   
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
