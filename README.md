**癫痫发作预测项目的复现方案**。以下是项目的高层次结构设计，包括数据处理、特征提取、模型训练和预测模块。该复现计划遵循论文中提出的方法：**数据增强 + CSP 特征提取 + 浅层 CNN + 卡尔曼滤波预测触发机制**。

---

## ✅ 项目复现整体思路（逐模块分解）

---

### 📁 1. 数据准备模块（CHB-MIT）

**目标**：从 [CHB-MIT dataset](https://physionet.org/files/chbmit/1.0.0/) 下载并整理患者 EEG 信号。

**步骤**：

* 使用 `mne` 或 `pyedflib` 读取 `.edf` 文件；
* 使用注释文件识别每个样本中的发作段（ictal）和预测段（pre-ictal）；
* 对每个样本进行切分，生成标签：

  * pre-ictal：发作前30分钟；
  * inter-ictal：远离发作时间段；
  * ictal：发作区段（训练中排除）；
* 每段信号切分为 **5秒片段（trial）**，形成 `X (channels×time)` + `y (标签)`。

---

### 🧹 2. 数据预处理模块

**目标**：对 EEG 信号进行滤波、统一通道选择和标准化处理。

**步骤**：

* 使用 Butterworth 滤波器（带通 5–50Hz）；
* 选取 18 个共同通道（FP1-F7, F7-T7, ...）；
* 每个 5s 信号标准化；
* 构建 `E ∈ ℝ^{N×P}`，其中 N 为通道数（18），P 为点数（例如 1280）。

---

### 🧠 3. 数据增强模块（pre-ictal 拼接）

**目标**：解决数据不平衡问题（pre-ictal 样本少）。

**策略**：

* 随机从不同 pre-ictal 片段中切分为 3 段；
* 将这些段拼接为新的样本；
* 同时丢弃部分 inter-ictal 片段，控制正负样本比例为 2:3（正:负）；
* 用于训练的数据集样本量得到扩充。

---

### 📈 4. 特征提取模块（CSP + 多频段 + 时序切分）

**目标**：将 EEG 信号转化为低维判别特征矩阵。

**方法**：

* 对每个 EEG 信号应用 **小波包分解** → 得到 8 个频带 + 原始信号（共 9）；
* 对每个频带 + 每个 trial：

  * 使用 CSP 提取前 m 行和后 m 行（通常 m = 3）；
  * 输出 `9 × 2m` 特征向量；
* 将 5s EEG trial 划分为两个 2.5s 小段；

  * 每段分别提取特征，组合为 `18×18` 特征矩阵。

---

### 🧰 5. 模型模块（浅层 CNN）

**目标**：构建二分类模型预测 trial 的状态（0: inter-ictal, 1: pre-ictal）。

**网络结构（论文复现）**：

* **输入**：`18×18` 特征矩阵；
* **Block1**：

  * Conv2D(kernel=3×3, filters=6) + LeakyReLU + MaxPool + Dropout(p=0.5)；
* **Block2**：

  * Conv2D(kernel=3×3, filters=16) + LeakyReLU + MaxPool + Dropout(p=0.5)；
* Flatten；
* FC1: Dense(84) + Sigmoid；
* FC2: Dense(2) + Softmax（输出 pre-ictal/inter-ictal）；
* Loss: CrossEntropy；
* Optimizer: Adam；
* Epoch-wise validation，防止过拟合。

---

### 🔁 6. 后处理模块（卡尔曼滤波 + 报警机制）

**目标**：平滑 CNN 输出，避免单个片段引发误报。

**机制**（论文中公式11）：

```math
f_out[n] = (1/T) * ∑_{k=n-T}^{n} O[k]
```

* 每次连续预测中，如果 `f_out[n] = 1`，则**触发报警**；
* 实验中 `T = 5`，即需要连续5段都为 pre-ictal 才触发。

---

### 📊 7. 性能评估指标

**输出指标**：

* Accuracy 准确率；
* Sensitivity 灵敏度；
* FPR 误报率（次/小时）；
* AUC；
* paired t-test 比较不同方法的显著性。

---

## ✅ 项目结构建议

```
epilepsy_prediction/
├── data/                        # EDF数据/预处理片段
├── utils/
│   ├── preprocessing.py         # 滤波、切片、通道对齐
│   ├── augment.py               # 数据增强逻辑
│   ├── csp.py                   # CSP特征提取
│   ├── wavelet.py               # 小波包分解
│   ├── kalman.py                # 卡尔曼滤波
├── models/
│   └── cnn.py                   # 浅层CNN结构定义
├── train.py                     # 训练主脚本
├── predict.py                   # 推理与报警
├── evaluate.py                  # 评估指标计算
├── config.yaml                  # 超参数配置
└── README.md
```

---

下一步建议我可以为你写出：

* ✅ `train.py` 主框架
* ✅ `cnn.py` 模型构建代码
* ✅ `csp.py` 特征提取器实现（NumPy）
  你希望我先从哪一部分开始写？或者是否先搭建一个最小可运行框架？
