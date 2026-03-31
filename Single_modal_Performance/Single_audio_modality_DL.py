import os
import time
import datetime
import copy
import pandas as pd
import numpy as np
import warnings
import gc

# 引入 compute_class_weight 解决类别不平衡
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

# ==========================================
# 0. 环境配置：警告静音与网络镜像源加速
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import timm
import soundfile as sf
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 手动 SEResNet50 权重路径 ======================
SERESNET50_WEIGHT_PATH = r"D:\Code\Project\Dataset\Models\seresnet50.ra2_in1k\pytorch_model.bin"


# ==========================================
# 模块 1：深度学习模型库
# ==========================================
class CustomCRNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))


def build_model(model_name, num_classes=3):
    if model_name == "ResNet18":
        model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    elif model_name == "ResNetSE":
        model = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000)
        state_dict = torch.load(SERESNET50_WEIGHT_PATH, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.reset_classifier(num_classes)
    elif model_name == "CRNN":
        model = CustomCRNN(num_classes=num_classes)
    elif model_name == "ViT (AST Surrogate)":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    return model


# ==========================================
# 模块 2：音频处理管道 (三划分适配 & ImageNet 对齐)
# ==========================================
class FixedAudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sr=16000, target_duration=5.0):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_mels=224, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # 【核心新增】：对齐预训练视觉骨干网络的 ImageNet 均值和方差
        self.imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform_np, sr = sf.read(file_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).float()

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

            num_samples = waveform.shape[1]
            if num_samples > self.target_samples:
                waveform = waveform[:, :self.target_samples]
            elif num_samples < self.target_samples:
                padding = self.target_samples - num_samples
                waveform = F.pad(waveform, (0, padding))

            mel_spec = self.mel_transform(waveform)
            mel_db = self.db_transform(mel_spec)

            # 实例级标准化 (屏蔽特定录音设备的全局音量差异)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

            # 转伪彩 3 通道
            mel_rgb = mel_db.repeat(3, 1, 1)

            # 【应用 ImageNet 归一化】
            mel_rgb = self.imagenet_normalize(mel_rgb)

            mel_final = F.interpolate(mel_rgb.unsqueeze(0), size=(224, 224), mode='bilinear',
                                      align_corners=False).squeeze(0)
            return mel_final, self.labels[idx]

        except Exception as e:
            print(f" [警告] 音频读取失败 {file_path}: {e}")
            return torch.zeros((3, 224, 224)), self.labels[idx]


def load_audio_dataset_from_csv(csv_path, audio_base_dir):
    print(f"\n[数据加载] 正在读取全局主划分名单: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, _, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith('.wav'):
                task_id = os.path.basename(root)
                path_mapping[task_id] = os.path.join(root, file)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

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

    print(f"[数据加载] 完毕！成功装载: 训练集 {len(train_paths)} | 验证集 {len(val_paths)} | 测试集 {len(test_paths)}")

    # 自动计算类权重解决不平衡问题
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"[数据分布] 自动计算训练集损失权重 (HC/MCI/AD): {class_weights_tensor.cpu().numpy()}")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_weights_tensor


# ==========================================
# 模块 3：严谨的盲测日志与模型调优
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
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

    # 【注入类权重】
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
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


def run_audio_dl_experiments(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
                             class_weights_tensor):
    print(f"\n[环境检查] 深度计算核心: {DEVICE.type.upper()}")

    # 构建懒加载 Dataset
    train_ds = FixedAudioDataset(train_paths, train_labels)
    val_ds = FixedAudioDataset(val_paths, val_labels)
    test_ds = FixedAudioDataset(test_paths, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model_names = ["ResNet18", "ResNetSE", "CRNN", "ViT (AST Surrogate)"]

    # 【分离超参池】保护巨型模型
    lr_list_cnn = [1e-3, 5e-4, 1e-4, 5e-5]
    lr_list_vit = [5e-5, 3e-5, 2e-5, 1e-5]
    WEIGHT_DECAY = 1e-4

    all_detailed_results = []

    print("\n" + "=" * 70)
    print("开始严谨调优 (每步表现计入CSV，仅最优模型接受测试集检验)")
    print("=" * 70)

    for name in model_names:
        print(f"\n>>> 调优模型: {name} <<<")
        is_vit = ("ViT" in name)
        current_lr_list = lr_list_vit if is_vit else lr_list_cnn

        # 动态控制收敛策略
        MAX_EPOCHS = 10 if is_vit else 50
        PATIENCE = 3 if is_vit else 7

        temp_logs = []

        # 1. 验证集上的盲测网格搜索
        for current_lr in current_lr_list:
            model = build_model(name, num_classes=3).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(
                model, train_loader, val_loader,
                epochs=MAX_EPOCHS, lr=current_lr, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, class_weights_tensor=class_weights_tensor
            )

            print(f"  [-] LR: {current_lr:<7} | 验证集 AUC: {val_auc:>6.2f}% (耗时: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Model": name,
                "Learning_Rate": current_lr,
                "Val_AUC": val_auc,
                "Weights_Dict": copy.deepcopy(best_wts)
            })

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        # 2. 锁定最优超参
        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(
            f"[!] {name} 调优结束 -> 最优 LR: {best_log['Learning_Rate']} (验证集最高 AUC: {best_log['Val_AUC']:.2f}%)")

        # 3. 终极测试 (唯一一次 Test Set 前向传播)
        print(f"[*] 正在解锁盲测集，进行唯一一次终极测试...")
        final_model = build_model(name, num_classes=3).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])

        test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
        print(f"    --> 测试集终极表现: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%\n")

        del final_model
        gc.collect()

        # 4. 汇总填表
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

    # 生成详细 DataFrame
    df_results = pd.DataFrame(all_detailed_results)

    # 生成供论文引用的 LaTeX (仅最优结果)
    df_optimal = df_results[df_results["Is_Optimal_Config"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Model']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Audio Deep Models Performance on Independent Test Set}\n\\label{tab:audio_dl_aligned}\n\\end{table}"

    return df_results, latex_str


# ==========================================
# 模块 4：执行入口
# ==========================================
if __name__ == "__main__":
    CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Single_modal_Performance\Single_audio_DL"

    print("正在测试 SEResNet50 权重是否能正常加载...")
    test_model = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000)
    state_dict = torch.load(SERESNET50_WEIGHT_PATH, map_location='cpu')
    test_model.load_state_dict(state_dict, strict=True)
    test_model.reset_classifier(3)
    print("SEResNet50 权重加载测试成功！\n")

    tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts = load_audio_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

    if len(tr_p) > 0 and len(ts_p) > 0 and len(val_p) > 0:
        df_results, latex_snippet = run_audio_dl_experiments(tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts)

        print("\n--- 供论文直接引用的 LaTeX 代码片段 (仅包含最优表现) ---")
        print(latex_snippet)
        print("--------------------------------------------------\n")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(OUTPUT_BASE_DIR, f"Audio_Opt_Aligned_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        csv_save_path = os.path.join(full_save_path, "detailed_grid_search_metrics.csv")
        df_results.to_csv(csv_save_path, index=False)

        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[执行完毕] 优化结果已安全保存至: {full_save_path}")
        print(f" -> 详细超参探索日志已保存为: detailed_grid_search_metrics.csv")
    else:
        print("[终止] 数据集提取失败，请检查 CSV 文件内容或音频源目录。")