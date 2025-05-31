import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
from models.CNN import SeizureCNN
import wandb
from pykalman import KalmanFilter
import torch.nn.functional as F
from utils.early_stop import EarlyStopping

def load_data(npz_path):
    data = np.load(npz_path)
    X = data['F']  # (N, 2, 36)
    y = data['y']
    X = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X).float().unsqueeze(1)  # (N, 1, 2, 36)
    y_tensor = torch.tensor(y).long()
    return X_tensor, y_tensor


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
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']

    return avg_loss, acc, current_lr

def kalman_smooth(pred_probs):
    """
    对预测概率进行卡尔曼滤波，输入为 (N,) 的 pre-ictal 概率序列。
    返回平滑后的序列。
    """
    kf = KalmanFilter(initial_state_mean=0.5, n_dim_obs=1)
    smoothed_state_means, _ = kf.smooth(pred_probs.reshape(-1, 1))
    return smoothed_state_means.ravel()

def evaluate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # 获取 pre-ictal 类别概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 卡尔曼滤波
    smoothed_probs = kalman_smooth(np.array(all_probs))
    pred_labels = (smoothed_probs > 0.5).astype(int)  # 阈值二分类

    # 准确率计算
    acc = (pred_labels == np.array(all_labels)).sum() / len(all_labels)
    return acc

def train_one_patient(args, input_file):
    print(f"\n🔍 正在处理文件: {input_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(input_file)
    dataset = TensorDataset(X, y)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    patient_val_acc = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n📂 Fold {fold+1}/{args.k_folds}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)

        model = SeizureCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=10)

        wandb.watch(model, log="all")
        
        for epoch in range(args.epochs):
            train_loss, train_acc, lr = train(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr
            })

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            scheduler.step()
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print("🛑 Early stopping triggered.")
                break

        patient_val_acc.append(val_acc)

    avg = np.mean(patient_val_acc)
    std = np.std(patient_val_acc)
    wandb.log({
        "patient_avg_val_accuracy": avg,
        "patient_val_std": std
    })
    return avg, std

def main(args):
    overall_acc = []
    wandb.init(
        project="seizure-prediction",
        config=vars(args),
        name="CrossPatientValidation"
    )

    for i in range(1, 24):  # chb01 to chb23
        subj_id = f"chb{i:02d}"
        input_file = os.path.join(args.input_dir, subj_id, "features.npz")
        if os.path.exists(input_file):
            acc, std = train_one_patient(args, input_file)
            overall_acc.append(acc)
        else:
            print(f"⚠️ 文件不存在: {input_file}")

    wandb.log({
        "final_avg_val_acc": np.mean(overall_acc),
        "final_std_val_acc": np.std(overall_acc)
    })
    print(f"\n✅ 所有患者的平均准确率：{np.mean(overall_acc):.4f} ± {np.std(overall_acc):.4f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeizureCNN with cross-validation.")
    parser.add_argument('--input_dir', type=str, default='data/features/', help='Directory containing .npz files')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()
    main(args)
