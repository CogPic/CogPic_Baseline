import os
import time
import datetime
import copy
import pandas as pd
import numpy as np
import warnings
import gc
from PIL import Image
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

# ==========================================
# 0. 环境警告与日志静音配置
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models.video as video_models
import timm
from torchvision import transforms
import torchaudio
import torchaudio.functional as F_audio
import soundfile as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 本地模型绝对路径配置 (请确保路径正确)
# ==========================================
TIMM_RESNET18_PATH = r"D:\Code\Project\Dataset\Models\timm_resnet18\pytorch_model.bin"
SERESNET50_WEIGHT_PATH = r"D:\Code\Project\Dataset\Models\seresnet50.ra2_in1k\pytorch_model.bin"
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Cross_modal_Performance\DL_Benchmark\Audio_Video_diff"


# ==========================================
# 模块 1：严格对齐的音频-视频数据加载
# ==========================================
def load_aligned_audio_video_dataset_from_csv(csv_path, base_dir):
    print(f"\n[阶段 1] 数据加载 -> 正在读取全局主划分名单: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, dirs, files in os.walk(base_dir):
        task_id = os.path.basename(root)
        has_wav = any(f.endswith('.wav') for f in files)
        has_video_dir = any('120' in d for d in dirs)

        if has_wav and has_video_dir:
            path_mapping[task_id] = root

    train_recs, val_recs, test_recs, train_labels = [], [], [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            task_dir = path_mapping[task_id]
            wav_file = next(f for f in os.listdir(task_dir) if f.endswith('.wav'))
            video_dir = next(d for d in os.listdir(task_dir) if '120' in d)

            record = {
                'wav_path': os.path.join(task_dir, wav_file),
                'video_path': os.path.join(task_dir, video_dir),
                'label': label_idx
            }

            if split_type == 'Train':
                train_recs.append(record)
                train_labels.append(label_idx)
            elif split_type == 'Validation':
                val_recs.append(record)
            elif split_type == 'Test':
                test_recs.append(record)

    print(f"  -> 成功装载视听双模态数据: 训练集 {len(train_recs)} | 验证集 {len(val_recs)} | 测试集 {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class AudioVideoAblationDataset(Dataset):
    def __init__(self, data_records, target_sr=16000, target_duration=5.0):
        self.data_records = data_records
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=224, n_fft=1024,
                                                                  hop_length=512)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.imagenet_normalize
        ])
        self.dummy_frame = self.video_transform(Image.new('RGB', (224, 224), (0, 0, 0)))

    def _process_audio(self, file_path):
        try:
            waveform_np, sr = sf.read(file_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.target_sr:
                waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

            num_samples = waveform.shape[1]
            if num_samples > self.target_samples:
                waveform = waveform[:, :self.target_samples]
            elif num_samples < self.target_samples:
                waveform = F.pad(waveform, (0, self.target_samples - num_samples))

            mel_db = self.db_transform(self.mel_transform(waveform))
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_rgb = self.imagenet_normalize(mel_db.repeat(3, 1, 1))
            return F.interpolate(mel_rgb.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        except:
            return torch.zeros((3, 224, 224))

    def _load_video(self, folder_path):
        pt_path = os.path.join(folder_path, "video_tensor.pt")
        if os.path.exists(pt_path):
            try:
                return torch.load(pt_path, map_location='cpu', weights_only=True)
            except:
                pass

        frames = []
        try:
            img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])[:120]
            for f in img_files:
                img = Image.open(os.path.join(folder_path, f)).convert('RGB')
                frames.append(self.video_transform(img))

            if len(frames) > 0:
                last_f = frames[-1]
                while len(frames) < 120: frames.append(last_f)
            else:
                while len(frames) < 120: frames.append(self.dummy_frame)
        except:
            frames = [self.dummy_frame] * 120

        return torch.stack(frames).permute(1, 0, 2, 3)  # 输出形状: (3, 120, 224, 224)

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        audio_tensor = self._process_audio(record['wav_path'])
        video_tensor = self._load_video(record['video_path'])
        return audio_tensor, video_tensor, record['label']


# ==========================================
# 模块 2：动态视听骨干网络工厂 (升级版 - 包含显存保护机制)
# ==========================================
class AudioCRNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.lstm = nn.LSTM(128, output_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out.mean(dim=1)


class AudioVideoFusionNet(nn.Module):
    def __init__(self, audio_model_name, video_model_name, num_classes=3, dropout_rate=0.5, freeze_backbones=True):
        super().__init__()
        self.audio_model_name = audio_model_name
        self.video_model_name = video_model_name
        self.freeze_backbones = freeze_backbones

        # --- A. 音频编码器 ---
        if audio_model_name == "ResNetSE":
            # 兼容原代码的本地 SEResNet50 离线加载方式
            print(f"  -> [权重加载] 正在严格加载本地 SEResNet50 (Audio): {SERESNET50_WEIGHT_PATH}")
            if not os.path.exists(SERESNET50_WEIGHT_PATH):
                raise FileNotFoundError("\n[致命错误] 未找到本地音频预训练权重，拒绝随机初始化！")
            self.audio_encoder = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000,
                                                   checkpoint_path=SERESNET50_WEIGHT_PATH)
            self.audio_encoder.reset_classifier(0)
            audio_dim = 2048  # SEResNet50 提取出的特征维度是 2048

        elif audio_model_name == "CRNN":
            self.audio_encoder = AudioCRNN(output_dim=256)
            audio_dim = 256
        else:
            raise ValueError(f"不支持的音频模型: {audio_model_name}")

        # --- B. 视频编码器 ---
        if video_model_name == "MC3_18":
            self.video_encoder = video_models.mc3_18(pretrained=True)
            self.video_encoder.fc = nn.Identity()
            video_dim = 512

        elif video_model_name == "R3D_18":
            self.video_encoder = video_models.r3d_18(pretrained=True)
            self.video_encoder.fc = nn.Identity()
            video_dim = 512
        else:
            raise ValueError(f"不支持的视频模型: {video_model_name}")

        # 【安全保护】：如果需要冻结，显式关闭参数的梯度属性
        if self.freeze_backbones:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.video_encoder.parameters():
                param.requires_grad = False

        # --- C. 融合分类头 (保持开启梯度) ---
        fusion_dim = audio_dim + video_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_input, video_input):
        B, C, T, H, W = video_input.size()
        if H != 112 or W != 112:
            video_input = F.interpolate(video_input, size=(T, 112, 112), mode='trilinear', align_corners=False)

        # 【显存保护】：彻底切断骨干网络的梯度图构建
        if self.freeze_backbones:
            with torch.no_grad():
                a_feat = self.audio_encoder(audio_input)
                v_feat = self.video_encoder(video_input)
        else:
            a_feat = self.audio_encoder(audio_input)
            v_feat = self.video_encoder(video_input)

        fused_feat = torch.cat((a_feat, v_feat), dim=1)
        logits = self.classifier(fused_feat)

        return logits


# ==========================================
# 模块 3：OOM 防御训练引擎 (含进度条与实时Loss)
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for audios, videos, labels in dataloader:
            audios, videos = audios.to(DEVICE), videos.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(audios, videos)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_labels.extend(labels.numpy())

    uar = recall_score(all_labels, all_preds, average='macro') * 100
    war = recall_score(all_labels, all_preds, average='weighted') * 100
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0
    return uar, war, auc


def train_eval_single_fold(model, train_loader, val_loader, class_weights_tensor, lr, epochs=20, patience=6,
                           accumulation_steps=8):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"LR: {lr:<7} | Epoch {epoch + 1}/{epochs}", leave=False, ncols=100)

        for i, (audios, videos, labels) in enumerate(pbar):
            audios, videos, labels = audios.to(DEVICE), videos.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(audios, videos)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

        _, _, val_auc = evaluate_model(model, val_loader)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    -> [早停机制触发] 验证集连续 {patience} 轮无提升，提前结束该学习率的训练。")
            break

    model.load_state_dict(best_model_wts)
    return best_val_auc, best_model_wts


# ==========================================
# 模块 4：执行视听双模态架构消融大满贯
# ==========================================
def run_audio_video_ablation(tr_recs, val_recs, ts_recs, class_wts):
    train_ds = AudioVideoAblationDataset(tr_recs)
    val_ds = AudioVideoAblationDataset(val_recs)
    test_ds = AudioVideoAblationDataset(ts_recs)

    # 应对双模态 3D 卷积的极限降压：物理 BatchSize=2，Accumulation=8，等效 16
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # 【核心修改点】：仅保留我们精选的两个王牌对照组合
    combinations = [
        {"audio": "ResNetSE", "video": "MC3_18", "paradigm": "Attention & Multi-scale Spatiotemporal"},
        {"audio": "CRNN", "video": "R3D_18", "paradigm": "Explicit Temporal & Standard 3D"}
    ]

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    all_results = []

    print("\n" + "=" * 80)
    print("开始 Audio+Video 视听联合提取消融实验 (冻结骨干网络，微调 Concat-MLP)")
    print("=" * 80)

    for combo in combinations:
        a_name, v_name, paradigm = combo["audio"], combo["video"], combo["paradigm"]
        print(f"\n>>> 正在验证范式: {paradigm} ({a_name} + {v_name}) <<<")
        temp_logs = []

        for lr in lr_list:
            # 实例化全新模型，默认开启 freeze_backbones=True 防爆显存
            model = AudioVideoFusionNet(a_name, v_name, num_classes=3, freeze_backbones=True).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(
                model, train_loader, val_loader, class_wts, lr,
                epochs=25, patience=6, accumulation_steps=8
            )

            print(f"  [-] LR: {lr:<7} | 验证集最佳 AUC: {val_auc:>6.2f}% (总耗时: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Paradigm": paradigm, "Audio_Encoder": a_name, "Video_Encoder": v_name,
                "Learning_Rate": lr, "Val_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
            })
            del model;
            torch.cuda.empty_cache();
            gc.collect()

        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(f"[!] {paradigm} 最优网络结构定型 (最佳LR: {best_log['Learning_Rate']})，即将执行仅此一次的测试集盲测...")

        # 测试阶段重新加载最优权重
        final_model = AudioVideoFusionNet(a_name, v_name, num_classes=3, freeze_backbones=True).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])
        test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
        print(f"    --> 测试集终极表现: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%")

        best_log["Test_UAR"] = test_uar
        best_log["Test_WAR"] = test_war
        best_log["Test_AUC"] = test_auc

        del final_model;
        gc.collect()

        for log in temp_logs:
            is_op = log["Learning_Rate"] == best_log["Learning_Rate"]
            all_results.append({
                "Paradigm": log["Paradigm"], "Audio_Encoder": log["Audio_Encoder"],
                "Video_Encoder": log["Video_Encoder"],
                "Learning_Rate": log["Learning_Rate"], "Val_AUC": log["Val_AUC"],
                "Test_UAR": log.get("Test_UAR") if is_op else None,
                "Test_WAR": log.get("Test_WAR") if is_op else None,
                "Test_AUC": log.get("Test_AUC") if is_op else None,
                "Is_Optimal": is_op
            })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
    latex_str += "\\textbf{Paradigm} & \\textbf{Audio + Video Encoders} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\midrule\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Paradigm']} & {row['Audio_Encoder']} + {row['Video_Encoder']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Ablation Study of Audio-Visual Feature Encoders}\n\\label{tab:audio_video_ablation}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    if len(os.listdir(DATASET_BASE_DIR)) > 0:
        tr_recs, val_recs, ts_recs, class_wts = load_aligned_audio_video_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

        if len(tr_recs) > 0:
            df_results, latex_snippet = run_audio_video_ablation(tr_recs, val_recs, ts_recs, class_wts)

            print("\n--- 供论文引用的 视听联合架构敏感性消融表 ---")
            print(latex_snippet)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
            df_results.to_csv(os.path.join(OUTPUT_BASE_DIR, f"Audio_Video_Ablation_Results_{current_time}.csv"),
                              index=False)
            with open(os.path.join(OUTPUT_BASE_DIR, f"latex_table_{current_time}.txt"), 'w', encoding='utf-8') as f:
                f.write(latex_snippet)