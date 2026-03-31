import os
import time
import datetime
import copy
import pandas as pd
import numpy as np
import warnings
import gc

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
import torchaudio
import torchaudio.functional as F_audio  # 【修正】：引入函数式 API
import soundfile as sf
import timm
import torchvision.transforms as transforms

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SERESNET50_WEIGHT_PATH = r"D:\Code\Project\Dataset\Models\seresnet50.ra2_in1k\pytorch_model.bin"


# ==========================================
# 模块 1：严格对齐的文本-音频数据加载
# ==========================================
def load_aligned_dataset_from_csv(csv_path, base_dir):
    print(f"\n[阶段 1] 数据加载 -> 正在读取全局主划分名单: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, dirs, files in os.walk(base_dir):
        task_id = os.path.basename(root)
        has_txt = any(f.endswith('.txt') for f in files)
        has_wav = any(f.endswith('.wav') for f in files)
        if has_txt and has_wav:
            path_mapping[task_id] = root

    train_recs, val_recs, test_recs, train_labels = [], [], [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            task_dir = path_mapping[task_id]
            txt_file = next(f for f in os.listdir(task_dir) if f.endswith('.txt'))
            wav_file = next(f for f in os.listdir(task_dir) if f.endswith('.wav'))

            record = {
                'txt_path': os.path.join(task_dir, txt_file),
                'wav_path': os.path.join(task_dir, wav_file),
                'label': label_idx
            }

            if split_type == 'Train':
                train_recs.append(record)
                train_labels.append(label_idx)
            elif split_type == 'Validation':
                val_recs.append(record)
            elif split_type == 'Test':
                test_recs.append(record)

    print(f"  -> 成功装载双模态配对数据: 训练集 {len(train_recs)} | 验证集 {len(val_recs)} | 测试集 {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class TextAudioAblationDataset(Dataset):
    def __init__(self, data_records, tokenizer, max_txt_len=256, target_sr=16000, target_duration=5.0):
        self.data_records = data_records
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=224, n_fft=1024,
                                                                  hop_length=512)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _read_text(self, path):
        # 【修正】：健壮的中文双编码降级读取
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='gbk') as f:
                    return f.read().strip()
            except Exception:
                return ""
        except Exception:
            return ""

    def _process_audio(self, file_path):
        try:
            waveform_np, sr = sf.read(file_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)

            # 【修正】：使用函数式 API 避免 DataLoader 内存泄漏和性能下降
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

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        text = self._read_text(record['txt_path'])
        encoded = self.tokenizer(text if text else "未知", padding='max_length', truncation=True,
                                 max_length=self.max_txt_len, return_tensors='pt')
        mel_tensor = self._process_audio(record['wav_path'])
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), mel_tensor, record['label']


# ==========================================
# 模块 2：动态骨干网络工厂 (严谨学术版)
# ==========================================
class AttentionBiLSTM(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        # 【修正】：移除内置 Embedding，防泄露
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_weights(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return context


class AudioCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out.mean(dim=1)


class BackboneAblationFusionNet(nn.Module):
    def __init__(self, text_model_name, audio_model_name, bert_path, num_classes=3):
        super().__init__()
        self.text_model_name = text_model_name
        self.audio_model_name = audio_model_name

        # 公用绝对冻结的静态 BERT
        self.frozen_bert = BertModel.from_pretrained(bert_path)
        for param in self.frozen_bert.parameters(): param.requires_grad = False

        # --- Text 分支 ---
        if text_model_name == "Att-BiLSTM":
            self.text_aggregator = AttentionBiLSTM(embed_dim=768, hidden_dim=128)
            text_dim = 256
        elif text_model_name == "BERT-base":
            self.text_aggregator = nn.Identity()
            text_dim = 768

        # --- Audio 分支 ---
        if audio_model_name == "CRNN":
            self.audio_encoder = AudioCRNN()
            audio_dim = 256

        elif audio_model_name == "ResNet18":
            # 【修正】：加入黄金标准对比基线
            self.audio_encoder = timm.create_model('resnet18', pretrained=True, num_classes=0)
            for param in self.audio_encoder.parameters(): param.requires_grad = False
            audio_dim = 512

        elif audio_model_name == "SEResNet50":
            self.audio_encoder = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=0)
            try:
                self.audio_encoder.load_state_dict(torch.load(SERESNET50_WEIGHT_PATH, map_location='cpu'), strict=False)
            except:
                pass
            for param in self.audio_encoder.parameters(): param.requires_grad = False
            audio_dim = 2048

        # --- 统一的 Concat-MLP 融合策略 ---
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, masks, mels):
        with torch.no_grad():
            bert_out = self.frozen_bert(input_ids, attention_mask=masks)

        if self.text_model_name == "Att-BiLSTM":
            t_feat = self.text_aggregator(bert_out.last_hidden_state)
        else:
            t_feat = bert_out.last_hidden_state[:, 0, :]

        if self.audio_model_name == "CRNN":
            a_feat = self.audio_encoder(mels)
        else:
            with torch.no_grad():
                a_feat = self.audio_encoder(mels)

        return self.classifier(torch.cat((t_feat, a_feat), dim=1))


# ==========================================
# 模块 3：严谨训练与测试引擎 (附带梯度累加)
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for ids, masks, mels, labels in dataloader:
            ids, masks, mels = ids.to(DEVICE), masks.to(DEVICE), mels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(ids, masks, mels)
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


def train_eval_single_fold(model, train_loader, val_loader, class_weights_tensor, lr, epochs=25, patience=6,
                           accumulation_steps=4):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for i, (ids, masks, mels, labels) in enumerate(train_loader):
            ids, masks, mels, labels = ids.to(DEVICE), masks.to(DEVICE), mels.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(ids, masks, mels)
                # 【修正】：执行安全的梯度缩放
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        _, _, val_auc = evaluate_model(model, val_loader)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience: break

    model.load_state_dict(best_model_wts)
    return best_val_auc, best_model_wts


# ==========================================
# 模块 4：执行大满贯消融实验
# ==========================================
def run_backbone_ablation(tr_recs, val_recs, ts_recs, class_wts, bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_ds = TextAudioAblationDataset(tr_recs, tokenizer)
    val_ds = TextAudioAblationDataset(val_recs, tokenizer)
    test_ds = TextAudioAblationDataset(ts_recs, tokenizer)

    # 物理 batch_size 配合梯度累加，兼顾大模型显存限制与平稳收敛
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    combinations = [
        {"text": "Att-BiLSTM", "audio": "CRNN", "paradigm": "Sequential Baseline"},
        {"text": "BERT-base", "audio": "ResNet18", "paradigm": "Standard Baseline"},
        {"text": "BERT-base", "audio": "SEResNet50", "paradigm": "Deep Attention"}
    ]

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    all_results = []

    print("\n" + "=" * 80)
    print("开始 Backbone 消融实验 (固化 Concat-MLP，探讨原始模态表征潜力)")
    print("=" * 80)

    for combo in combinations:
        t_name, a_name, paradigm = combo["text"], combo["audio"], combo["paradigm"]
        print(f"\n>>> 正在验证范式: {paradigm} ({t_name} + {a_name}) <<<")
        temp_logs = []

        for lr in lr_list:
            model = BackboneAblationFusionNet(t_name, a_name, bert_path).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(
                model, train_loader, val_loader, class_wts, lr,
                epochs=25, patience=6, accumulation_steps=4
            )

            print(f"  [-] LR: {lr:<7} | 验证集 AUC: {val_auc:>6.2f}% (耗时: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Paradigm": paradigm, "Text_Encoder": t_name, "Audio_Encoder": a_name,
                "Learning_Rate": lr, "Val_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
            })
            del model;
            torch.cuda.empty_cache();
            gc.collect()

        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(f"[!] {paradigm} 最优网络结构定型，即将执行测试集盲测...")

        final_model = BackboneAblationFusionNet(t_name, a_name, bert_path).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])
        test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
        print(f"    --> 测试集终极表现: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%")

        # 【修正】：将测试指标直接且唯一地挂载到最优日志字典上，消除作用域隐患
        best_log["Test_UAR"] = test_uar
        best_log["Test_WAR"] = test_war
        best_log["Test_AUC"] = test_auc

        del final_model;
        gc.collect()

        for log in temp_logs:
            is_op = (log["Learning_Rate"] == best_log["Learning_Rate"])
            all_results.append({
                "Paradigm": log["Paradigm"], "Text_Encoder": log["Text_Encoder"], "Audio_Encoder": log["Audio_Encoder"],
                "Learning_Rate": log["Learning_Rate"], "Val_AUC": log["Val_AUC"],
                "Test_UAR": log.get("Test_UAR") if is_op else None,
                "Test_WAR": log.get("Test_WAR") if is_op else None,
                "Test_AUC": log.get("Test_AUC") if is_op else None,
                "Is_Optimal": is_op
            })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
    latex_str += "\\textbf{Paradigm} & \\textbf{Text + Audio Encoders} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\midrule\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Paradigm']} & {row['Text_Encoder']} + {row['Audio_Encoder']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Ablation Study of Feature Encoders with Fixed Concat-MLP Fusion}\n\\label{tab:backbone_ablation}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Cross_modal_Performance\DL_Benchmark\Text_Audio_diff"
    BERT_PATH = r"D:\Code\Project\Dataset\Models\bert-base-chinese"

    tr_recs, val_recs, ts_recs, class_wts = load_aligned_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

    if len(tr_recs) > 0:
        df_results, latex_snippet = run_backbone_ablation(tr_recs, val_recs, ts_recs, class_wts, BERT_PATH)

        print("\n--- 供论文引用的 Backbone 敏感性消融表 ---")
        print(latex_snippet)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        df_results.to_csv(os.path.join(OUTPUT_BASE_DIR, f"Ablation_Results_{current_time}.csv"), index=False)
        with open(os.path.join(OUTPUT_BASE_DIR, f"latex_table_{current_time}.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)