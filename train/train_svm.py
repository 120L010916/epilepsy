from collections import defaultdict
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
# import wandb
from sklearn.model_selection import KFold
from utils.train_utils import  load_data, kalman_smooth, evaluate_metrics, EarlyStopping
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
# os.environ['WANDB_MODE'] = 'disabled'

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1) #每个样本预测的类别索引
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']

    return avg_loss, acc, current_lr


def evaluate(model, dataloader, device, use_kalman=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 只取每个样本属于类别1的概率值（即正类概率）。
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            if not use_kalman:
                preds = torch.argmax(outputs, dim=1)
                y_pred.extend(preds.cpu().numpy())
                
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # 🎯 卡尔曼滤波（可选）
    if use_kalman:
        y_pred = kalman_smooth(y_prob)
    else:
        # 二值预测
        # y_pred = (y_prob > 0.5).astype(int)
        # _, y_pred = torch.max(outputs.data, 1)
        y_pred = np.array(y_pred)

    return y_true, y_pred, y_prob

def train_one_patient(args, input_file, target_val_acc=0.80, max_epochs=200):
    print(f"\n🔍 正在处理文件: {input_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(input_file)
    X = X.view(X.shape[0], -1).numpy()
    y = y.numpy()
    use_kalman = True
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n📂 Fold {fold+1}/{args.k_folds} (K-Fold)")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        if use_kalman:
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # 避免除 0
            best_idx = f1_scores.argmax()
            best_threshold = thresholds[best_idx]
            y_pred = kalman_smooth(y_prob, threshold=best_threshold, T=5)
        
        
        metrics = evaluate_metrics(y_test, y_pred, y_prob)
        
        # 如果需要卡尔曼滤波，可以在这里应用
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        metrics_list.append(metrics)

    # 汇总所有 fold 的平均结果
    avg_metrics = {
        f"avg_{k}": np.mean([m[k] for m in metrics_list])
        for k in metrics_list[0]
    }

    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    return avg_metrics


def main(args):
    all_avg_metrics = defaultdict(list)  # 用于存储每个患者的平均 metrics
    
    # wandb.init(
    #     project="seizure-prediction",
    #     config=vars(args),
    #     name="CrossPatientValidation"
    # )

    for i in range(1, 24):  # chb01 to chb23
        subj_id = f"chb{i:02d}"
        input_file = os.path.join(args.input_dir, subj_id, "features.npz")
        if os.path.exists(input_file):
            avg_metrics = train_one_patient(args, input_file)
            for k, v in avg_metrics.items():
                all_avg_metrics[k].append(v)
        else:
            print(f"⚠️ 文件不存在: {input_file}")

    final_metrics = {f"final_avg_{k}": np.mean(vs) for k, vs in all_avg_metrics.items()}
    # wandb.log({
    #     **final_metrics
    # })
    print("📊 所有患者平均指标：")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    # wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeizureCNN with cross-validation.")
    parser.add_argument('--input_dir', type=str, default='data/features/', help='Directory containing .npz files')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--k_folds', type=int, default=2, help='Number of folds for K-Fold cross-validation')
    
    args = parser.parse_args()
    main(args)
