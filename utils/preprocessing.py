import mne
from datetime import timedelta
import pandas as pd



def load_edf_mne(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.set_eeg_reference('average', projection=True)  # 可选：均值参考
    return raw



def load_seizure_annotations(annotation_file):
    # 假设注释为 CSV 格式：[patient, record, seizure_start, seizure_end]
    return pd.read_csv(annotation_file)



def get_labeled_segments(duration, seizure_times, pre_ictal_minutes=30):
    label_map = []
    used_ranges = []
    
    for start, end in seizure_times:
        # 1. ictal 区段
        label_map.append(('ictal', start, end))
        used_ranges.append((start, end))
        
        # 2. pre-ictal 区段
        pre_start = max(0, start - pre_ictal_minutes * 60)
        label_map.append(('pre-ictal', pre_start, start))
        used_ranges.append((pre_start, start))
    
    # 3. inter-ictal 区段：避开发作段±1小时
    from_end = 0
    for label, start, end in sorted(used_ranges):
        if from_end < start - 3600:
            label_map.append(('inter-ictal', from_end, start - 3600))
        from_end = max(from_end, end + 3600)

    if from_end < duration:
        label_map.append(('inter-ictal', from_end, duration))

    return label_map


import numpy as np

def segment_trials(raw, label_map, trial_length=5):
    X, y = [], []
    sfreq = int(raw.info['sfreq'])
    data = raw.get_data()

    for label, start, end in label_map:
        if label == 'ictal':  # 可排除 ictal 段用于训练
            continue

        start_idx = int(start * sfreq)
        end_idx = int(end * sfreq)
        step = trial_length * sfreq

        for i in range(start_idx, end_idx - step, step):
            trial = data[:, i:i + step]
            if trial.shape[1] == step:
                X.append(trial)
                y.append(label)
    
    return np.array(X), np.array(y)



def preprocess_data(file_path, annotation_file, pre_ictal_minutes=30, trial_length=5):
    raw = load_edf_mne(file_path)
    duration = raw.times[-1]  # 获取数据总时长（秒）

    annotations = load_seizure_annotations(annotation_file)
    seizure_times = [(row['seizure_start'], row['seizure_end']) for _, row in annotations.iterrows()]

    label_map = get_labeled_segments(duration, seizure_times, pre_ictal_minutes)

    X, y = segment_trials(raw, label_map, trial_length)

    return X, y

def save_preprocessed_data(X, y, output_file):
    import joblib
    data = {'X': X, 'y': y}
    joblib.dump(data, output_file)
    print(f"Data saved to {output_file}")
    
def load_preprocessed_data(input_file):
    import joblib
    data = joblib.load(input_file)
    X, y = data['X'], data['y']
    print(f"Data loaded from {input_file}")
    return X, y








