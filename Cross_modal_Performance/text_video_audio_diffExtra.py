import os
import time
import datetime
import copy
import argparse
import pandas as pd
import numpy as np
import warnings
import gc
from PIL import Image
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

# ==========================================
# 0. Environment & Logging Setup
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Module 1: Tri-Modal Data Loading (Task-Aware)
# ==========================================
def load_trimodal_dataset_from_csv(csv_path, base_dir):
    print(f"\n[Data Loading] Reading master split list: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, dirs, files in os.walk(base_dir):
        task_id = os.path.basename(root)
        has_txt = any(f.endswith('.txt') for f in files)
        has_wav = any(f.endswith('.wav') for f in files)
        has_video_dir = any('120' in d for d in dirs)

        if has_txt and has_wav and has_video_dir:
            path_mapping[task_id] = root

    train_recs, val_recs, test_recs, train_labels = [], [], [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        task_type = "Unknown"
        if task_id.endswith('001'):
            task_type = "Pic 1"
        elif task_id.endswith('002'):
            task_type = "Pic 2"
        elif task_id.endswith('003'):
            task_type = "Pic 3"

        if task_id in path_mapping and task_type != "Unknown":
            task_dir = path_mapping[task_id]
            txt_file = next(f for f in os.listdir(task_dir) if f.endswith('.txt'))
            wav_file = next(f for f in os.listdir(task_dir) if f.endswith('.wav'))
            video_dir = next(d for d in os.listdir(task_dir) if '120' in d)

            record = {
                'txt_path': os.path.join(task_dir, txt_file),
                'wav_path': os.path.join(task_dir, wav_file),
                'video_path': os.path.join(task_dir, video_dir),
                'label': label_idx,
                'task_type': task_type
            }

            if split_type == 'Train':
                train_recs.append(record)
                train_labels.append(label_idx)
            elif split_type == 'Validation':
                val_recs.append(record)
            elif split_type == 'Test':
                test_recs.append(record)

    print(
        f"  -> Successfully loaded tri-modal data: Train {len(train_recs)} | Val {len(val_recs)} | Test {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class TriModalAblationDataset(Dataset):
    def __init__(self, data_records, tokenizer, max_txt_len=256, target_sr=16000, target_duration=5.0):
        self.data_records = data_records
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        self.target_sr, self.target_samples = target_sr, int(target_sr * target_duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_mels=224, n_fft=1024, hop_length=512)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), self.imagenet_normalize
        ])
        self.dummy_frame = self.video_transform(Image.new('RGB', (224, 224), (0, 0, 0)))
        self.task_map = {"Pic 1": 0, "Pic 2": 1, "Pic 3": 2}

    def _read_text(self, path):
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
            if sr != self.target_sr: waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

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
            for f in img_files: frames.append(
                self.video_transform(Image.open(os.path.join(folder_path, f)).convert('RGB')))
            if len(frames) > 0:
                last_f = frames[-1]
                while len(frames) < 120: frames.append(last_f)
            else:
                while len(frames) < 120: frames.append(self.dummy_frame)
        except:
            frames = [self.dummy_frame] * 120

        return torch.stack(frames).permute(1, 0, 2, 3)

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        text = self._read_text(record['txt_path'])
        encoded = self.tokenizer(text if text else "Unknown", padding='max_length', truncation=True,
                                 max_length=self.max_txt_len, return_tensors='pt')
        audio_tensor = self._process_audio(record['wav_path'])
        video_tensor = self._load_video(record['video_path'])
        task_idx = self.task_map[record['task_type']]
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), audio_tensor, video_tensor, \
            record['label'], task_idx


# ==========================================
# Module 2: Tri-Modal Backbone Factory
# ==========================================
class AttentionBiLSTM(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_weights(out), dim=1)
        return torch.sum(attn_weights * out, dim=1)


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


class ResNetLSTM_Video(nn.Module):
    def __init__(self, freeze_cnn=True, weight_path=None):
        super().__init__()
        if weight_path and os.path.exists(weight_path):
            print(f"  -> [Weights] Loading local ResNet18 (Video): {weight_path}")
            backbone = timm.create_model('resnet18', pretrained=False, num_classes=1000, checkpoint_path=weight_path)
        else:
            print("  -> [Weights] Downloading default ResNet18 from timm.")
            backbone = timm.create_model('resnet18', pretrained=True, num_classes=1000)

        backbone.reset_classifier(0)
        self.cnn = backbone
        if freeze_cnn:
            for p in self.cnn.parameters(): p.requires_grad = False
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)
        out, _ = self.lstm(features)
        return out.mean(dim=1)


class TriModalAblationFusionNet(nn.Module):
    def __init__(self, text_model_name, audio_model_name, video_model_name, bert_path, num_classes=3, resnet_path=None,
                 seresnet_path=None):
        super().__init__()
        self.text_model_name, self.audio_model_name, self.video_model_name = text_model_name, audio_model_name, video_model_name

        self.frozen_bert = BertModel.from_pretrained(bert_path)
        for param in self.frozen_bert.parameters(): param.requires_grad = False

        # === 1. Text ===
        if text_model_name == "Att-BiLSTM":
            self.text_aggregator = AttentionBiLSTM(embed_dim=768, hidden_dim=128)
            text_dim = 256
        elif text_model_name == "BERT-base":
            self.text_aggregator = nn.Identity()
            text_dim = 768

        # === 2. Audio ===
        if audio_model_name == "CRNN":
            self.audio_encoder = AudioCRNN()
            audio_dim = 256
        elif audio_model_name == "ResNet18":
            if resnet_path and os.path.exists(resnet_path):
                self.audio_encoder = timm.create_model('resnet18', pretrained=False, num_classes=1000,
                                                       checkpoint_path=resnet_path)
            else:
                self.audio_encoder = timm.create_model('resnet18', pretrained=True, num_classes=1000)
            self.audio_encoder.reset_classifier(0)
            for param in self.audio_encoder.parameters(): param.requires_grad = False
            audio_dim = 512
        elif audio_model_name == "SEResNet50":
            if seresnet_path and os.path.exists(seresnet_path):
                self.audio_encoder = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000,
                                                       checkpoint_path=seresnet_path)
            else:
                self.audio_encoder = timm.create_model('seresnet50.ra2_in1k', pretrained=True, num_classes=1000)
            self.audio_encoder.reset_classifier(0)
            for param in self.audio_encoder.parameters(): param.requires_grad = False
            audio_dim = 2048

        # === 3. Video ===
        if video_model_name == "ResNet+LSTM":
            self.video_encoder = ResNetLSTM_Video(freeze_cnn=True, weight_path=resnet_path)
            video_dim = 512
        elif video_model_name == "R3D_18":
            self.video_encoder = video_models.r3d_18(pretrained=True)
            self.video_encoder.fc = nn.Identity()
            for param in self.video_encoder.parameters(): param.requires_grad = False
            video_dim = 512
        elif video_model_name == "R2Plus1D":
            self.video_encoder = video_models.r2plus1d_18(pretrained=True)
            self.video_encoder.fc = nn.Identity()
            for param in self.video_encoder.parameters(): param.requires_grad = False
            video_dim = 512

        total_dim = text_dim + audio_dim + video_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, ids, masks, audios, videos):
        with torch.no_grad(): bert_out = self.frozen_bert(ids, attention_mask=masks)
        t_feat = self.text_aggregator(
            bert_out.last_hidden_state) if self.text_model_name == "Att-BiLSTM" else bert_out.last_hidden_state[:, 0, :]

        a_feat = self.audio_encoder(audios) if self.audio_model_name == "CRNN" else self.audio_encoder(audios).detach()
        v_feat = self.video_encoder(videos) if self.video_model_name == "ResNet+LSTM" else self.video_encoder(
            videos).detach()

        return self.classifier(torch.cat((t_feat, a_feat, v_feat), dim=1))


# ==========================================
# Module 3: Task-Aware Evaluation & Engine
# ==========================================
def safe_metrics(y_true, y_pred, y_prob):
    if len(y_true) == 0: return 0.0, 0.0, 0.0
    uar = recall_score(y_true, y_pred, average='macro') * 100
    war = recall_score(y_true, y_pred, average='weighted') * 100
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0
    return uar, war, auc


def evaluate_multitask_model(model, dataloader, desc="Evaluating"):
    model.eval()
    eval_pools = {
        "Global": {"probs": [], "preds": [], "labels": []},
        "Pic 1": {"probs": [], "preds": [], "labels": []},
        "Pic 2": {"probs": [], "preds": [], "labels": []},
        "Pic 3": {"probs": [], "preds": [], "labels": []}
    }
    idx_to_task = {0: "Pic 1", 1: "Pic 2", 2: "Pic 3"}

    pbar_eval = tqdm(dataloader, desc=f"  -> {desc}", leave=False, ncols=100)

    with torch.no_grad():
        for ids, masks, audios, videos, labels, task_idxs in pbar_eval:
            ids, masks, audios, videos = ids.to(DEVICE), masks.to(DEVICE), audios.to(DEVICE), videos.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(ids, masks, audios, videos)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            preds, labels_np, tasks_np = np.argmax(probs, axis=1), labels.numpy(), task_idxs.numpy()

            for i in range(len(labels_np)):
                t_name = idx_to_task[tasks_np[i]]
                eval_pools["Global"]["probs"].append(probs[i]);
                eval_pools["Global"]["preds"].append(preds[i]);
                eval_pools["Global"]["labels"].append(labels_np[i])
                eval_pools[t_name]["probs"].append(probs[i]);
                eval_pools[t_name]["preds"].append(preds[i]);
                eval_pools[t_name]["labels"].append(labels_np[i])

    results = {}
    for pool_name, pool_data in eval_pools.items():
        uar, war, auc = safe_metrics(pool_data["labels"], pool_data["preds"], pool_data["probs"])
        results[pool_name] = {"UAR": uar, "WAR": war, "AUC": auc}
    return results


def train_eval_single_fold(model, train_loader, val_loader, class_weights_tensor, lr, epochs=25, patience=6,
                           accumulation_steps=8):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler('cuda')

    best_val_global_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"LR: {lr:<7} | Epoch {epoch + 1}/{epochs}", leave=False, ncols=100)

        for i, (ids, masks, audios, videos, labels, _) in enumerate(pbar):
            ids, masks, audios, videos, labels = ids.to(DEVICE), masks.to(DEVICE), audios.to(DEVICE), videos.to(
                DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(ids, masks, audios, videos)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

        val_res = evaluate_multitask_model(model, val_loader, desc=f"Validating Epoch {epoch + 1}")
        global_auc = val_res["Global"]["AUC"]

        if global_auc > best_val_global_auc:
            best_val_global_auc = global_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    -> [Early Stopping] Global AUC showed no improvement for {patience} epochs. Terminating LR.")
            break

    model.load_state_dict(best_model_wts)
    return best_val_global_auc, best_model_wts


# ==========================================
# Module 4: Master Execution Engine
# ==========================================
def run_trimodal_backbone_ablation(tr_recs, val_recs, ts_recs, class_wts, bert_path, resnet_path, seresnet_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_ds = TriModalAblationDataset(tr_recs, tokenizer)
    val_ds = TriModalAblationDataset(val_recs, tokenizer)
    test_ds = TriModalAblationDataset(ts_recs, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    combinations = [
        {"text": "Att-BiLSTM", "audio": "CRNN", "video": "ResNet+LSTM", "paradigm": "Sequential Decoupled"},
        {"text": "BERT-base", "audio": "ResNet18", "video": "R3D_18", "paradigm": "Joint Spatiotemporal Baseline"},
        {"text": "BERT-base", "audio": "SEResNet50", "video": "R2Plus1D", "paradigm": "Deep Attention & Factorized"}
    ]

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    all_results = []

    print("\n" + "=" * 90)
    print("Starting T+A+V Tri-Modal Ablation Study (With Multi-Task Perception)")
    print("=" * 90)

    for combo in combinations:
        t_name, a_name, v_name, paradigm = combo["text"], combo["audio"], combo["video"], combo["paradigm"]
        print(f"\n>>> Validating Paradigm: {paradigm} <<<")
        print(f"    [T]: {t_name} | [A]: {a_name} | [V]: {v_name}")
        temp_logs = []

        for lr in lr_list:
            model = TriModalAblationFusionNet(t_name, a_name, v_name, bert_path, resnet_path=resnet_path,
                                              seresnet_path=seresnet_path).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(model, train_loader, val_loader, class_wts, lr, epochs=25,
                                                       patience=6, accumulation_steps=8)
            print(f"  [-] LR: {lr:<7} | Best Val Global AUC: {val_auc:>6.2f}% (Time: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Paradigm": paradigm, "Encoders": f"{t_name}+{a_name}+{v_name}",
                "Learning_Rate": lr, "Val_Global_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
            })
            del model
            torch.cuda.empty_cache()
            gc.collect()

        best_log = max(temp_logs, key=lambda x: x["Val_Global_AUC"])
        print(f"[!] Optimal configuration acquired (Best LR: {best_log['Learning_Rate']}). Unlocking blind test set...")

        final_model = TriModalAblationFusionNet(t_name, a_name, v_name, bert_path, resnet_path=resnet_path,
                                                seresnet_path=seresnet_path).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])
        test_res = evaluate_multitask_model(final_model, test_loader, desc="Final Blind Testing")

        g_res = test_res["Global"]
        print(
            f"    --> Final Global Performance: AUC {g_res['AUC']:.2f}% | UAR {g_res['UAR']:.2f}% | WAR {g_res['WAR']:.2f}%")

        best_log.update({
            "Test_Global_UAR": test_res["Global"]["UAR"], "Test_Global_WAR": test_res["Global"]["WAR"],
            "Test_Global_AUC": test_res["Global"]["AUC"],
            "Test_Pic1_UAR": test_res["Pic 1"]["UAR"], "Test_Pic1_WAR": test_res["Pic 1"]["WAR"],
            "Test_Pic1_AUC": test_res["Pic 1"]["AUC"],
            "Test_Pic2_UAR": test_res["Pic 2"]["UAR"], "Test_Pic2_WAR": test_res["Pic 2"]["WAR"],
            "Test_Pic2_AUC": test_res["Pic 2"]["AUC"],
            "Test_Pic3_UAR": test_res["Pic 3"]["UAR"], "Test_Pic3_WAR": test_res["Pic 3"]["WAR"],
            "Test_Pic3_AUC": test_res["Pic 3"]["AUC"]
        })

        del final_model
        gc.collect()

        for log in temp_logs:
            is_op = (log["Learning_Rate"] == best_log["Learning_Rate"])
            all_results.append({
                "Paradigm": log["Paradigm"], "Encoders": log["Encoders"], "Learning_Rate": log["Learning_Rate"],
                "Val_Global_AUC": log["Val_Global_AUC"],
                "Test_Global_UAR": log.get("Test_Global_UAR") if is_op else None,
                "Test_Global_WAR": log.get("Test_Global_WAR") if is_op else None,
                "Test_Global_AUC": log.get("Test_Global_AUC") if is_op else None,
                "Test_Pic1_UAR": log.get("Test_Pic1_UAR") if is_op else None,
                "Test_Pic1_WAR": log.get("Test_Pic1_WAR") if is_op else None,
                "Test_Pic1_AUC": log.get("Test_Pic1_AUC") if is_op else None,
                "Test_Pic2_UAR": log.get("Test_Pic2_UAR") if is_op else None,
                "Test_Pic2_WAR": log.get("Test_Pic2_WAR") if is_op else None,
                "Test_Pic2_AUC": log.get("Test_Pic2_AUC") if is_op else None,
                "Test_Pic3_UAR": log.get("Test_Pic3_UAR") if is_op else None,
                "Test_Pic3_WAR": log.get("Test_Pic3_WAR") if is_op else None,
                "Test_Pic3_AUC": log.get("Test_Pic3_AUC") if is_op else None,
                "Is_Optimal": is_op
            })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    latex_str = "\\begin{table*}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lll ccc ccc ccc ccc}\n\\toprule\n"
    latex_str += "\\multirow{2}{*}{\\textbf{Paradigm}} & \\multirow{2}{*}{\\textbf{Encoders (T+A+V)}} & \\multirow{2}{*}{\\textbf{Best LR}} & \\multicolumn{3}{c}{\\textbf{Global}} & \\multicolumn{3}{c}{\\textbf{Pic. 1}} & \\multicolumn{3}{c}{\\textbf{Pic. 2}} & \\multicolumn{3}{c}{\\textbf{Pic. 3}} \\\\\n"
    latex_str += "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-12} \\cmidrule(lr){13-15}\n"
    latex_str += " & & & \\textbf{UAR} & \\textbf{WAR} & \\textbf{AUC} & \\textbf{UAR} & \\textbf{WAR} & \\textbf{AUC} & \\textbf{UAR} & \\textbf{WAR} & \\textbf{AUC} & \\textbf{UAR} & \\textbf{WAR} & \\textbf{AUC} \\\\\n\\midrule\n"

    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Paradigm']} & {row['Encoders']} & {row['Learning_Rate']:.0e} & "
        latex_str += f"{row['Test_Global_UAR']:.2f} & {row['Test_Global_WAR']:.2f} & {row['Test_Global_AUC']:.2f} & "
        latex_str += f"{row['Test_Pic1_UAR']:.2f} & {row['Test_Pic1_WAR']:.2f} & {row['Test_Pic1_AUC']:.2f} & "
        latex_str += f"{row['Test_Pic2_UAR']:.2f} & {row['Test_Pic2_WAR']:.2f} & {row['Test_Pic2_AUC']:.2f} & "
        latex_str += f"{row['Test_Pic3_UAR']:.2f} & {row['Test_Pic3_WAR']:.2f} & {row['Test_Pic3_AUC']:.2f} \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Ablation Study of Tri-Modal Encoders Across Different Picture Tasks}\n\\label{tab:trimodal_backbone_ablation}\n\\end{table*}"

    return df_results, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-Modal Baseline Ablation Pipeline")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Cross_modal/TriModal_Baselines',
                        help='Output directory')
    parser.add_argument('--bert_path', type=str, default='bert-base-chinese', help='Path to BERT model')
    parser.add_argument('--resnet_path', type=str, default=None, help='Local path for ResNet18 weights (optional)')
    parser.add_argument('--seresnet_path', type=str, default=None, help='Local path for SEResNet50 weights (optional)')

    args = parser.parse_args()

    if len(os.listdir(args.data_dir)) > 0:
        tr_recs, val_recs, ts_recs, class_wts = load_trimodal_dataset_from_csv(args.csv_path, args.data_dir)

        if len(tr_recs) > 0:
            df_results, latex_snippet = run_trimodal_backbone_ablation(
                tr_recs, val_recs, ts_recs, class_wts, args.bert_path, args.resnet_path, args.seresnet_path)

            print("\n--- LaTeX Table Snippet for Citation ---")
            print(latex_snippet)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.output_dir, exist_ok=True)
            df_results.to_csv(os.path.join(args.output_dir, f"TriModal_Ablation_Results_{current_time}.csv"),
                              index=False)
            with open(os.path.join(args.output_dir, f"latex_table_{current_time}.txt"), 'w', encoding='utf-8') as f:
                f.write(latex_snippet)