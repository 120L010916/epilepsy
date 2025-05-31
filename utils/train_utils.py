from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score, confusion_matrix, cohen_kappa_score
)
import numpy as np
import torch
import torch.nn as nn
from pykalman import KalmanFilter


def load_data(npz_path):
    data = np.load(npz_path)
    X = data['F']  # (N, 2, 36)
    y = data['y']
    X = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X).float().unsqueeze(1)  # (N, 1, 2, 36)
    y_tensor = torch.tensor(y).long()
    return X_tensor, y_tensor

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def evaluate_metrics(y_true, y_pred, y_prob=None):
    """
    y_true: np.array of true labels (0 or 1)
    y_pred: np.array of predicted labels (0 or 1)
    y_prob: np.array of predicted probabilities for positive class (optional, for AUC)
    """
    metrics = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 灵敏度（Sensitivity） = TP / (TP + FN)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 误报率（FPR） = FP / (FP + TN)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # 召回率 = sensitivity
    metrics['recall'] = recall_score(y_true, y_pred)

    # F1 值
    metrics['f1'] = f1_score(y_true, y_pred)

    # Kappa
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    # AUC（需要概率）
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0

    return metrics
       
def kalman_smooth(pred_probs):
    """
    对预测概率进行卡尔曼滤波，输入为 (N,) 的 pre-ictal 概率序列。
    返回平滑后的序列。
    """
    kf = KalmanFilter(initial_state_mean=0.5, n_dim_obs=1)
    smoothed_state_means, _ = kf.smooth(pred_probs.reshape(-1, 1))
    return smoothed_state_means.ravel()    




       