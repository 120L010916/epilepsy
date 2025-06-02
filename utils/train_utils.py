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
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Sensitivity (Recall)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # total_time = len(y_true) * 5 # 总样本数 * 5秒
    # metrics['fpr'] = fp * 3600 / total_time
    # Accuracy
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Recall (same as sensitivity)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Cohen's Kappa
    try:
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        metrics['kappa'] = 0.0

    # AUC
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0

    return metrics

def kalman_smooth(pred_probs, threshold=0.5, T=5):
    """
    pred_probs: numpy array of predicted probabilities (for class 1: pre-ictal)
    threshold: threshold for binarizing predictions (default=0.8)
    T: smoothing window size (default=5)
    
    Returns:
        alarm_output: binary array with smoothed predictions using sliding window
    """
    # Step 1: Convert prob to binary prediction
    bin_pred = (pred_probs >= threshold).astype(int)

    # Step 2: Apply sliding window average
    alarm_output = np.zeros_like(bin_pred)

    for i in range(T - 1, len(bin_pred)):
        window = bin_pred[i - T + 1:i + 1]
        if np.sum(window) == T:  # 所有 T 个窗口都为 1，则触发报警
            alarm_output[i] = 1

    return alarm_output




       