import os
import time
import datetime
import copy
import pandas as pd
import numpy as np
import warnings
import gc
import contextlib

# 解决类别不平衡
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import soundfile as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 模块 1：纯波形端到端深度学习模型
# ==========================================
class RawWaveLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(RawWaveLSTM, self).__init__()

        # 1. 前端 1D-CNN 降采样模块
        # 输入 shape: (Batch, 1, 80000)
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 80000 -> 2500

            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 2500 -> 156

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU()
        )
        # 经过 frontend 后，输出 shape: (Batch, 256, 156)

        # 2. 时序建模 LSTM
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )

        # 3. 分类头
        self.fc = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = x.permute(0, 2, 1)  # (Batch, 156, 256)
        lstm_out, _ = self.lstm(x)
        pooled_out = lstm_out.mean(dim=1)
        return self.fc(pooled_out)


# ==========================================
# 模块 2：原始波形处理管道
# ==========================================
class RawAudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sr=16000, target_duration=5.0):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform_np, sr = sf.read(file_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).float()

            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            elif waveform.ndim == 0:
                waveform = waveform.unsqueeze(0)

            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

            num_samples = waveform.shape[0]
            if num_samples > self.target_samples:
                waveform = waveform[:self.target_samples]
            elif num_samples < self.target_samples:
                padding = self.target_samples - num_samples
                waveform = F.pad(waveform, (0, padding))

            # Z-score normalization 消除绝对音量差异
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
            waveform = waveform.unsqueeze(0)  # (1, 80000)

            return waveform, self.labels[idx]
        except Exception as e:
            print(f" [警告] 音频读取失败 {file_path}: {e}")
            return torch.zeros((1, self.target_samples)), self.labels[idx]


def load_audio_dataset_from_csv(csv_path, audio_base_dir):
    print(f"\n[数据加载] 正在读取全局主划分名单: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, _, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith('.wav'):
                task_id = os.path.basename(root)
                path_mapping[task_id] = os.path.join(root, file)

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = [], [], [], [], [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            file_path = path_mapping[task_id]
            if split_type == 'Train':
                train_paths.append(file_path)
                train_labels.append(label_idx)
            elif split_type == 'Validation':
                val_paths.append(file_path)
                val_labels.append(label_idx)
            elif split_type == 'Test':
                test_paths.append(file_path)
                test_labels.append(label_idx)

    print(f"[数据加载] 完毕！训练集 {len(train_paths)} | 验证集 {len(val_paths)} | 测试集 {len(test_paths)}")

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_weights_tensor


# ==========================================
# 模块 3：严谨的盲测日志与模型调优 (引入 AMP)
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            # 推理阶段同样可以使用 AMP 提升速度
            with torch.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else contextlib.nullcontext():
                logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_labels.extend(targets.numpy())

    uar = recall_score(all_labels, all_preds, average='macro') * 100
    war = recall_score(all_labels, all_preds, average='weighted') * 100
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except ValueError:
        auc = 0.0
    return uar, war, auc


def train_eval_single_fold(model, train_loader, val_loader, epochs, lr, weight_decay, patience, class_weights_tensor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 【核心新增】初始化 GradScaler (仅在 CUDA 环境下生效)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # 【核心新增】使用 AMP (自动混合精度) 前向传播
            with torch.autocast(device_type=DEVICE.type) if DEVICE.type == 'cuda' else contextlib.nullcontext():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if scaler is not None:
                # AMP 反向传播
                scaler.scale(loss).backward()

                # 【逻辑修正】在 clip 之前，必须先 unscale 梯度，否则裁剪阈值会失效
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新权重并更新 scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # 退回常规流程 (CPU 等环境)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        _, _, val_auc = evaluate_model(model, val_loader)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_auc, best_model_wts


def run_raw_wave_experiment(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_wts):
    print(f"\n[环境检查] 深度计算核心: {DEVICE.type.upper()}")

    train_ds = RawAudioDataset(train_paths, train_labels)
    val_ds = RawAudioDataset(val_paths, val_labels)
    test_ds = RawAudioDataset(test_paths, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    lr_list = [1e-3, 5e-4, 1e-4, 5e-5]
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 7

    temp_logs = []
    all_detailed_results = []

    print("\n" + "=" * 70)
    print("开始端到端波形模型 (RawWave-LSTM) 严谨调优 (AMP 加持)")
    print("=" * 70)

    for current_lr in lr_list:
        model = RawWaveLSTM(num_classes=3).to(DEVICE)
        start_t = time.time()

        val_auc, best_wts = train_eval_single_fold(
            model, train_loader, val_loader,
            epochs=MAX_EPOCHS, lr=current_lr, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, class_weights_tensor=class_wts
        )

        print(f"  [-] LR: {current_lr:<7} | 验证集 AUC: {val_auc:>6.2f}% (耗时: {time.time() - start_t:>2.0f}s)")

        temp_logs.append({
            "Model": "RawWave-LSTM",
            "Learning_Rate": current_lr,
            "Val_AUC": val_auc,
            "Weights_Dict": copy.deepcopy(best_wts)
        })

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
    print(f"\n[!] 调优结束 -> 最优 LR: {best_log['Learning_Rate']} (验证集最高 AUC: {best_log['Val_AUC']:.2f}%)")

    print(f"[*] 正在解锁盲测集，进行终极检验...")
    final_model = RawWaveLSTM(num_classes=3).to(DEVICE)
    final_model.load_state_dict(best_log['Weights_Dict'])

    test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
    print(f"    --> 测试集终极表现: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%\n")

    for log in temp_logs:
        is_optimal = (log["Learning_Rate"] == best_log["Learning_Rate"])
        all_detailed_results.append({
            "Model": log["Model"],
            "Learning_Rate": log["Learning_Rate"],
            "Val_AUC": log["Val_AUC"],
            "Test_UAR": test_uar if is_optimal else None,
            "Test_WAR": test_war if is_optimal else None,
            "Test_AUC": test_auc if is_optimal else None,
            "Is_Optimal_Config": is_optimal
        })

    df_results = pd.DataFrame(all_detailed_results)
    df_optimal = df_results[df_results["Is_Optimal_Config"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Model']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{End-to-End Raw Waveform Model Performance}\n\\label{tab:raw_wave_dl}\n\\end{table}"

    return df_results, latex_str


# ==========================================
# 模块 4：执行入口
# ==========================================
if __name__ == "__main__":
    CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Single_modal_Performance\RawWave_LSTM_Experiment"

    tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts = load_audio_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

    if len(tr_p) > 0 and len(ts_p) > 0 and len(val_p) > 0:
        df_results, latex_snippet = run_raw_wave_experiment(tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts)

        print("\n--- 供论文引用的 LaTeX 代码片段 ---")
        print(latex_snippet)
        print("----------------------------------\n")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(OUTPUT_BASE_DIR, f"RawWave_Opt_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        df_results.to_csv(os.path.join(full_save_path, "detailed_grid_search_metrics.csv"), index=False)
        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"[执行完毕] 优化结果已安全保存至: {full_save_path}")
    else:
        print("[终止] 数据集提取失败，请检查 CSV 文件内容或音频源目录。")