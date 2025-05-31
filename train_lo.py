from collections import defaultdict
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from models.CNN import SeizureCNN
import wandb
from sklearn.model_selection import LeaveOneOut

import torch.nn.functional as F
from utils.train_utils import  load_data, kalman_smooth, evaluate_metrics


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


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)



def train_one_patient(args, input_file):
    print(f"\n🔍 正在处理文件: {input_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(input_file)  # 假设返回的是 seizure-level 样本

    dataset = TensorDataset(X, y)
    loo = LeaveOneOut()

    val_accuracies = []
    metrics_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(dataset)):
        print(f"\n📂 Fold {fold+1}/{len(dataset)} (Leave-One-Out)")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size)

        model = SeizureCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        criterion = nn.CrossEntropyLoss()
        
        wandb.watch(model, log="all", log_freq=10)

        for epoch in range(args.epochs):
            train_loss, train_acc, lr = train(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            y_true, y_pred, y_prob = evaluate(model, val_loader, device)
            metrics = evaluate_metrics(y_true, y_pred, y_prob)

            # 打印
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            # 记录到 wandb
            wandb.log(metrics)

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        val_accuracies.append(val_acc)
        metrics_list.append(metrics)
        
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in metrics_list])
    avg = np.mean(val_accuracies)
    std = np.std(val_accuracies)

    wandb.log({
        **avg_metrics,
        "chb_avg_val_accuracy": avg,
        "chb_val_std": std
    })

    print(f"✅ LOOCV 平均准确率：{avg:.4f} ± {std:.4f}")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
        
    return avg, std, avg_metrics

def main(args):
    overall_acc = []
    all_avg_metrics = defaultdict(list)  # 用于存储每个患者的平均 metrics
    
    wandb.init(
        project="seizure-prediction",
        config=vars(args),
        name="CrossPatientValidation"
    )

    for i in range(1, 24):  # chb01 to chb23
        subj_id = f"chb{i:02d}"
        input_file = os.path.join(args.input_dir, subj_id, "features.npz")
        if os.path.exists(input_file):
            acc, std, avg_metrics = train_one_patient(args, input_file)
            overall_acc.append(acc)
            for k, v in avg_metrics.items():
                all_avg_metrics[k].append(v)
        else:
            print(f"⚠️ 文件不存在: {input_file}")

    final_metrics = {f"final_avg_{k}": np.mean(vs) for k, vs in all_avg_metrics.items()}
    wandb.log({
        "final_avg_val_acc": np.mean(overall_acc),
        "final_std_val_acc": np.std(overall_acc),
        **final_metrics
    })
    print(f"\n✅ 所有患者的平均准确率：{np.mean(overall_acc):.4f} ± {np.std(overall_acc):.4f}")
    print("📊 所有患者平均指标：")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeizureCNN with cross-validation.")
    parser.add_argument('--input_dir', type=str, default='data/features/', help='Directory containing .npz files')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
