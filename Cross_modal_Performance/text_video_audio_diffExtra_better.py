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

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 本地模型绝对路径配置 (请确保配置正确)
# ==========================================
SERESNET50_WEIGHT_PATH = r"D:\Code\Project\Dataset\Models\seresnet50.ra2_in1k\pytorch_model.bin"
BERT_PATH = r"D:\Code\Project\Dataset\Models\bert-base-chinese"
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
# 输出目录已更改为专属融合实验目录
OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Cross_modal_Performance\DL_Benchmark\Text_Video_Audio_diff\TextCNN_MC3_SEResNet_Fusion"


# ==========================================
# 模块 1：三模态数据加载 (包含多任务感知)
# ==========================================
def load_trimodal_dataset_from_csv(csv_path, base_dir):
    print(f"\n[阶段 1] 数据加载 -> 正在读取全局主划分名单: {csv_path}")
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

    print(f"  -> 成功装载三模态配对数据: 训练集 {len(train_recs)} | 验证集 {len(val_recs)} | 测试集 {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class TriModalDataset(Dataset):
    def __init__(self, data_records, tokenizer, max_txt_len=256, target_sr=16000, target_duration=5.0):
        self.data_records = data_records
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        self.target_sr, self.target_samples = target_sr, int(target_sr * target_duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=224, n_fft=1024,
                                                                  hop_length=512)
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
        encoded = self.tokenizer(text if text else "未知", padding='max_length', truncation=True,
                                 max_length=self.max_txt_len, return_tensors='pt')
        audio_tensor = self._process_audio(record['wav_path'])
        video_tensor = self._load_video(record['video_path'])
        task_idx = self.task_map[record['task_type']]
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), audio_tensor, video_tensor, \
        record['label'], task_idx


# ==========================================
# 模块 2：优选特征提取器网络 (去除分类头)
# ==========================================
class TextCNNFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=768, num_filters=100, filter_sizes=(3, 4, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        self.feature_dim = len(filter_sizes) * num_filters  # 300

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return torch.cat(x, 1)


class OptimalTriModalFusionNet(nn.Module):
    def __init__(self, bert_path, num_classes=3):
        super().__init__()
        print("  -> [架构加载] TextCNN (Text) + SEResNet50 (Audio) + MC3_18 (Video)")

        # 1. Text Modality (BERT Base + TextCNN)
        self.frozen_bert = BertModel.from_pretrained(bert_path)
        for param in self.frozen_bert.parameters(): param.requires_grad = False
        self.text_extractor = TextCNNFeatureExtractor(embedding_dim=768)
        text_dim = 300

        # 2. Audio Modality (SEResNet50)
        if not os.path.exists(SERESNET50_WEIGHT_PATH): raise FileNotFoundError(
            "\n[致命错误] 本地 SEResNet50 预训练权重缺失！")
        self.audio_extractor = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000)
        self.audio_extractor.load_state_dict(torch.load(SERESNET50_WEIGHT_PATH, map_location='cpu'), strict=True)
        self.audio_extractor.reset_classifier(0)  # 剔除最后的全连接层
        for param in self.audio_extractor.parameters(): param.requires_grad = False  # 冻结音频网络防过拟合
        audio_dim = 2048

        # 3. Video Modality (MC3_18)
        self.video_extractor = video_models.mc3_18(pretrained=True)
        self.video_extractor.fc = nn.Identity()  # 剔除最后的全连接层
        for param in self.video_extractor.parameters(): param.requires_grad = False  # 冻结视频网络防过拟合
        video_dim = 512

        # 4. Fusion Classifier (MLP)
        total_dim = text_dim + audio_dim + video_dim  # 2860
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, ids, masks, audios, videos):
        # 文本前向传播 (支持梯度的 TextCNN)
        with torch.no_grad():
            bert_out = self.frozen_bert(ids, attention_mask=masks).last_hidden_state
        t_feat = self.text_extractor(bert_out)

        # 音频、视频前向传播 (无梯度，纯提特征)
        with torch.no_grad():
            a_feat = self.audio_extractor(audios)
            v_feat = self.video_extractor(videos)

        return self.classifier(torch.cat((t_feat, a_feat, v_feat), dim=1))


# ==========================================
# 模块 3：评估引擎与训练循环 (保持原样，支持多任务 15 列指标感知)
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
            print(f"    -> [早停触发] Global AUC 连续 {patience} 轮无提升，终止当前 LR。")
            break

    model.load_state_dict(best_model_wts)
    return best_val_global_auc, best_model_wts


# ==========================================
# 模块 4：优选架构执行与导出大满贯引擎
# ==========================================
def run_optimal_fusion_experiment(tr_recs, val_recs, ts_recs, class_wts, bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_ds = TriModalDataset(tr_recs, tokenizer)
    val_ds = TriModalDataset(val_recs, tokenizer)
    test_ds = TriModalDataset(ts_recs, tokenizer)

    # 批次调小以防御 OOM，使用梯度累积补偿
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    temp_logs = []

    paradigm_name = "Optimal Late Fusion"
    encoders_name = "TextCNN+SEResNet50+MC3_18"

    print("\n" + "=" * 90)
    print(f"开始优选三模态融合盲测: {encoders_name}")
    print("=" * 90)

    for lr in lr_list:
        model = OptimalTriModalFusionNet(bert_path).to(DEVICE)
        start_t = time.time()

        val_auc, best_wts = train_eval_single_fold(
            model, train_loader, val_loader, class_wts, lr,
            epochs=25, patience=6, accumulation_steps=8
        )
        print(f"  [-] LR: {lr:<7} | 验证集 Global AUC: {val_auc:>6.2f}% (总耗时: {time.time() - start_t:>2.0f}s)")

        temp_logs.append({
            "Paradigm": paradigm_name, "Encoders": encoders_name,
            "Learning_Rate": lr, "Val_Global_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
        })
        del model
        torch.cuda.empty_cache()
        gc.collect()

    best_log = max(temp_logs, key=lambda x: x["Val_Global_AUC"])
    print(f"\n[!] 融合网络调优结束 (最佳LR: {best_log['Learning_Rate']})，解锁独立测试集执行终极盲测...")

    final_model = OptimalTriModalFusionNet(bert_path).to(DEVICE)
    final_model.load_state_dict(best_log['Weights_Dict'])
    test_res = evaluate_multitask_model(final_model, test_loader, desc="Final Blind Testing")

    g_res = test_res["Global"]
    print(f"    --> 全局终极表现: AUC {g_res['AUC']:.2f}% | UAR {g_res['UAR']:.2f}% | WAR {g_res['WAR']:.2f}%")

    all_results = []
    for log in temp_logs:
        is_op = (log["Learning_Rate"] == best_log["Learning_Rate"])
        all_results.append({
            "Paradigm": log["Paradigm"], "Encoders": log["Encoders"], "Learning_Rate": log["Learning_Rate"],
            "Val_Global_AUC": log["Val_Global_AUC"],
            "Test_Global_UAR": test_res["Global"]["UAR"] if is_op else None,
            "Test_Global_WAR": test_res["Global"]["WAR"] if is_op else None,
            "Test_Global_AUC": test_res["Global"]["AUC"] if is_op else None,
            "Test_Pic1_UAR": test_res["Pic 1"]["UAR"] if is_op else None,
            "Test_Pic1_WAR": test_res["Pic 1"]["WAR"] if is_op else None,
            "Test_Pic1_AUC": test_res["Pic 1"]["AUC"] if is_op else None,
            "Test_Pic2_UAR": test_res["Pic 2"]["UAR"] if is_op else None,
            "Test_Pic2_WAR": test_res["Pic 2"]["WAR"] if is_op else None,
            "Test_Pic2_AUC": test_res["Pic 2"]["AUC"] if is_op else None,
            "Test_Pic3_UAR": test_res["Pic 3"]["UAR"] if is_op else None,
            "Test_Pic3_WAR": test_res["Pic 3"]["WAR"] if is_op else None,
            "Test_Pic3_AUC": test_res["Pic 3"]["AUC"] if is_op else None,
            "Is_Optimal": is_op
        })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    # ================= 输出 15 列航母级 LaTeX 表格 =================
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

    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Performance of the Optimal Tri-Modal Late Fusion Configuration}\n\\label{tab:optimal_trimodal_fusion}\n\\end{table*}"

    return df_results, latex_str


if __name__ == "__main__":
    if len(os.listdir(DATASET_BASE_DIR)) > 0:
        tr_recs, val_recs, ts_recs, class_wts = load_trimodal_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

        if len(tr_recs) > 0:
            df_results, latex_snippet = run_optimal_fusion_experiment(tr_recs, val_recs, ts_recs, class_wts, BERT_PATH)

            print("\n--- 供论文引用的 15列超宽多任务感知表 ---")
            print(latex_snippet)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
            df_results.to_csv(os.path.join(OUTPUT_BASE_DIR, f"Optimal_Fusion_Results_{current_time}.csv"), index=False)
            with open(os.path.join(OUTPUT_BASE_DIR, f"latex_table_{current_time}.txt"), 'w', encoding='utf-8') as f:
                f.write(latex_snippet)

            print(f"\n[运行完毕] 详细结果与用于提交的 LaTeX 代码已存档至: {OUTPUT_BASE_DIR}")