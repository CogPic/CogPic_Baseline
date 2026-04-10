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
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Uncomment for mainland China users

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models.video as video_models
import timm
from torchvision import transforms

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Module 1: Strictly Aligned Data Loading
# ==========================================
def load_aligned_video_text_dataset_from_csv(csv_path, base_dir):
    print(f"\n[Data Loading] Reading master split list: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, dirs, files in os.walk(base_dir):
        task_id = os.path.basename(root)
        has_txt = any(f.endswith('.txt') for f in files)
        has_video_dir = any('120' in d for d in dirs)

        if has_txt and has_video_dir:
            path_mapping[task_id] = root

    train_recs, val_recs, test_recs, train_labels = [], [], [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            task_dir = path_mapping[task_id]
            txt_file = next(f for f in os.listdir(task_dir) if f.endswith('.txt'))
            video_dir = next(d for d in os.listdir(task_dir) if '120' in d)

            record = {
                'txt_path': os.path.join(task_dir, txt_file),
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

    print(
        f"  -> Successfully loaded paired bimodal data: Train {len(train_recs)} | Val {len(val_recs)} | Test {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class TextVideoAblationDataset(Dataset):
    def __init__(self, data_records, tokenizer, max_txt_len=256):
        self.data_records = data_records
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dummy_frame = self.video_transform(Image.new('RGB', (224, 224), (0, 0, 0)))

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

        return torch.stack(frames).permute(1, 0, 2, 3)

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        text = self._read_text(record['txt_path'])
        encoded = self.tokenizer(text if text else "Unknown", padding='max_length', truncation=True,
                                 max_length=self.max_txt_len, return_tensors='pt')
        video_tensor = self._load_video(record['video_path'])
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), video_tensor, record['label']


# ==========================================
# Module 2: Dynamic Spatiotemporal Backbone Factory
# ==========================================
class AttentionBiLSTM(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_weights(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return context


class ResNetLSTM_Video(nn.Module):
    def __init__(self, freeze_cnn=True, weight_path=None):
        super().__init__()

        if weight_path and os.path.exists(weight_path):
            backbone = timm.create_model('resnet18', pretrained=False, num_classes=1000, checkpoint_path=weight_path)
        else:
            backbone = timm.create_model('resnet18', pretrained=True, num_classes=1000)

        backbone.reset_classifier(0)
        self.cnn = backbone

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)
        out, _ = self.lstm(features)
        return out.mean(dim=1)


class VideoTextAblationFusionNet(nn.Module):
    def __init__(self, text_model_name, video_model_name, bert_path, num_classes=3, resnet_weight=None):
        super().__init__()
        self.text_model_name = text_model_name
        self.video_model_name = video_model_name

        self.frozen_bert = BertModel.from_pretrained(bert_path)
        for param in self.frozen_bert.parameters(): param.requires_grad = False

        # --- Text Encoder Pool ---
        if text_model_name == "Att-BiLSTM":
            self.text_aggregator = AttentionBiLSTM(embed_dim=768, hidden_dim=128)
            text_dim = 256
        elif text_model_name == "BERT-base":
            self.text_aggregator = nn.Identity()
            text_dim = 768

        # --- Video Spatiotemporal Encoder Pool ---
        if video_model_name == "ResNet+LSTM":
            self.video_encoder = ResNetLSTM_Video(freeze_cnn=True, weight_path=resnet_weight)
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

        # --- Base Feature Fusion Layer (Concat-MLP) ---
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + video_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, masks, video_tensor):
        with torch.no_grad():
            bert_out = self.frozen_bert(input_ids, attention_mask=masks)

        if self.text_model_name == "Att-BiLSTM":
            t_feat = self.text_aggregator(bert_out.last_hidden_state)
        else:
            t_feat = bert_out.last_hidden_state[:, 0, :]

        if self.video_model_name == "ResNet+LSTM":
            v_feat = self.video_encoder(video_tensor)
        else:
            with torch.no_grad():
                v_feat = self.video_encoder(video_tensor)

        return self.classifier(torch.cat((t_feat, v_feat), dim=1))


# ==========================================
# Module 3: Anti-OOM Training Engine
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for ids, masks, videos, labels in dataloader:
            ids, masks, videos = ids.to(DEVICE), masks.to(DEVICE), videos.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(ids, masks, videos)
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

        for i, (ids, masks, videos, labels) in enumerate(pbar):
            ids, masks, videos, labels = ids.to(DEVICE), masks.to(DEVICE), videos.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(ids, masks, videos)
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
            print(f"    -> [Early Stopping] Validation set showed no improvement for {patience} consecutive epochs.")
            break

    model.load_state_dict(best_model_wts)
    return best_val_auc, best_model_wts


# ==========================================
# Module 4: Execute Spatiotemporal Ablation
# ==========================================
def run_video_backbone_ablation(tr_recs, val_recs, ts_recs, class_wts, bert_path, resnet_weight):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_ds = TextVideoAblationDataset(tr_recs, tokenizer)
    val_ds = TextVideoAblationDataset(val_recs, tokenizer)
    test_ds = TextVideoAblationDataset(ts_recs, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    combinations = [
        {"text": "Att-BiLSTM", "video": "ResNet+LSTM", "paradigm": "Sequential Decoupled"},
        {"text": "BERT-base", "video": "R3D_18", "paradigm": "Joint Spatiotemporal"},
        {"text": "BERT-base", "video": "R2Plus1D", "paradigm": "Advanced R(2+1)D"}
    ]

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    all_results = []

    print("\n" + "=" * 80)
    print("Starting Text+Video Spatiotemporal Encoder Ablation Study")
    print("=" * 80)

    for combo in combinations:
        t_name, v_name, paradigm = combo["text"], combo["video"], combo["paradigm"]
        print(f"\n>>> Validating Paradigm: {paradigm} ({t_name} + {v_name}) <<<")
        temp_logs = []

        for lr in lr_list:
            model = VideoTextAblationFusionNet(t_name, v_name, bert_path, resnet_weight=resnet_weight).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(
                model, train_loader, val_loader, class_wts, lr,
                epochs=25, patience=6, accumulation_steps=8
            )

            print(f"  [-] LR: {lr:<7} | Best Val AUC: {val_auc:>6.2f}% (Time: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Paradigm": paradigm, "Text_Encoder": t_name, "Video_Encoder": v_name,
                "Learning_Rate": lr, "Val_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
            })
            del model
            torch.cuda.empty_cache()
            gc.collect()

        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(
            f"[!] Optimal configuration acquired for {paradigm} (Best LR: {best_log['Learning_Rate']}). Unlocking test set...")

        final_model = VideoTextAblationFusionNet(t_name, v_name, bert_path, resnet_weight=resnet_weight).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])
        test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
        print(f"    --> Final Test Performance: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%")

        best_log["Test_UAR"] = test_uar
        best_log["Test_WAR"] = test_war
        best_log["Test_AUC"] = test_auc

        del final_model
        gc.collect()

        for log in temp_logs:
            is_op = log["Learning_Rate"] == best_log["Learning_Rate"]
            all_results.append({
                "Paradigm": log["Paradigm"], "Text_Encoder": log["Text_Encoder"], "Video_Encoder": log["Video_Encoder"],
                "Learning_Rate": log["Learning_Rate"], "Val_AUC": log["Val_AUC"],
                "Test_UAR": log.get("Test_UAR") if is_op else None,
                "Test_WAR": log.get("Test_WAR") if is_op else None,
                "Test_AUC": log.get("Test_AUC") if is_op else None,
                "Is_Optimal": is_op
            })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
    latex_str += "\\textbf{Paradigm} & \\textbf{Text + Video Encoders} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\midrule\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Paradigm']} & {row['Text_Encoder']} + {row['Video_Encoder']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Ablation Study of Spatiotemporal Feature Encoders (Fixed Concat-MLP)}\n\\label{tab:video_backbone_ablation}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-Video Spatiotemporal Ablation Experiments")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/DL_Benchmark/Text_Video_diff',
                        help='Output directory')
    parser.add_argument('--bert_path', type=str, default='bert-base-chinese', help='Path to BERT model')
    parser.add_argument('--resnet_weight', type=str, default=None,
                        help='Path to local ResNet18 timm weights (optional)')

    args = parser.parse_args()

    if len(os.listdir(args.data_dir)) > 0:
        tr_recs, val_recs, ts_recs, class_wts = load_aligned_video_text_dataset_from_csv(args.csv_path, args.data_dir)

        if len(tr_recs) > 0:
            df_results, latex_snippet = run_video_backbone_ablation(tr_recs, val_recs, ts_recs, class_wts,
                                                                    args.bert_path, args.resnet_weight)

            print("\n--- LaTeX Table Snippet for Spatiotemporal Ablation ---")
            print(latex_snippet)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.output_dir, exist_ok=True)
            df_results.to_csv(os.path.join(args.output_dir, f"Video_Ablation_Results_{current_time}.csv"), index=False)
            with open(os.path.join(args.output_dir, f"latex_table_{current_time}.txt"), 'w', encoding='utf-8') as f:
                f.write(latex_snippet)