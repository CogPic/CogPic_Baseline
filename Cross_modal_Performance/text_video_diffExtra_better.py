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
from torchvision import transforms

from transformers import BertTokenizer, BertModel
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 本地模型绝对路径配置 (请确保路径正确)
# ==========================================
BERT_PATH = r"D:\Code\Project\Dataset\Models\bert-base-chinese"
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Cross_modal_Performance\DL_Benchmark\Text_Video_diff"


# ==========================================
# 模块 1：严格对齐的文本-视频数据加载
# ==========================================
def load_aligned_video_text_dataset_from_csv(csv_path, base_dir):
    print(f"\n[阶段 1] 数据加载 -> 正在读取全局主划分名单: {csv_path}")
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

    print(f"  -> 成功装载双模态配对数据: 训练集 {len(train_recs)} | 验证集 {len(val_recs)} | 测试集 {len(test_recs)}")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_recs, val_recs, test_recs, class_weights_tensor


class TextVideoDataset(Dataset):
    def __init__(self, data_records, tokenizer, max_txt_len=256):
        self.data_records = data_records
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        # 【保持原样】: 继续使用 224x224 保证与原有 R3D_18, ResNet+LSTM 消融实验的绝对公平对比
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
        encoded = self.tokenizer(text if text else "未知", padding='max_length', truncation=True,
                                 max_length=self.max_txt_len, return_tensors='pt')
        video_tensor = self._load_video(record['video_path'])
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), video_tensor, record['label']


# ==========================================
# 模块 2：专属提取器设计 (TextCNN + MC3_18)
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, embed_dim=768, num_filters=128, filter_sizes=(2, 3, 4)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])

    def forward(self, x):
        # x 形状: [Batch, SeqLen, EmbedDim] -> [Batch, 1, SeqLen, EmbedDim]
        x = x.unsqueeze(1)
        # 卷积 + ReLU + Squeeze
        out = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # Max-Pooling 提取最显著特征
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        # 拼接不同尺度卷积核的输出 (输出维度: 128 * 3 = 384)
        return torch.cat(out, 1)


class BestFusionNet(nn.Module):
    def __init__(self, bert_path, num_classes=3):
        super().__init__()

        # --- 冻结的 BERT 基座 ---
        self.frozen_bert = BertModel.from_pretrained(bert_path)
        for param in self.frozen_bert.parameters(): param.requires_grad = False

        # --- Text 编码器 (TextCNN) ---
        self.text_aggregator = TextCNN(embed_dim=768, num_filters=128, filter_sizes=(2, 3, 4))
        text_dim = 384

        # --- Video 编码器 (MC3_18) ---
        self.video_encoder = video_models.mc3_18(pretrained=True)
        self.video_encoder.fc = nn.Identity()
        for param in self.video_encoder.parameters(): param.requires_grad = False
        video_dim = 512

        # --- 固定的基准特征融合层 (Concat-MLP) ---
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + video_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, masks, video_tensor):
        # 1. 获取 BERT 的特征序列
        with torch.no_grad():
            bert_out = self.frozen_bert(input_ids, attention_mask=masks)

            # 【核心修改】: Attention Mask 清洗逻辑，过滤掉 padding 的噪声干扰
            expanded_mask = masks.unsqueeze(-1).float()
            masked_embeddings = bert_out.last_hidden_state * expanded_mask

        # 2. TextCNN 提取文本局部特征 (传入清洗后的 embeddings)
        t_feat = self.text_aggregator(masked_embeddings)

        # 3. MC3_18 提取视频时空特征
        with torch.no_grad():
            v_feat = self.video_encoder(video_tensor)

        # 4. 特征拼接并送入分类器
        return self.classifier(torch.cat((t_feat, v_feat), dim=1))


# ==========================================
# 模块 3：OOM 防御训练引擎与进度条监控
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


def train_eval_single_fold(model, train_loader, val_loader, class_weights_tensor, lr, epochs=25, patience=6,
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
            print(f"    -> [早停机制触发] 验证集连续 {patience} 轮无提升，提前结束该学习率的训练。")
            break

    model.load_state_dict(best_model_wts)
    return best_val_auc, best_model_wts


# ==========================================
# 模块 4：执行最优单模态结合测试
# ==========================================
def run_best_model_combination(tr_recs, val_recs, ts_recs, class_wts, bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_ds = TextVideoDataset(tr_recs, tokenizer)
    val_ds = TextVideoDataset(val_recs, tokenizer)
    test_ds = TextVideoDataset(ts_recs, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    lr_list = [5e-4, 2e-4, 1e-4, 5e-5]
    all_results = []
    temp_logs = []

    print("\n" + "=" * 80)
    print("开始执行最强单模态特征提取消融: TextCNN + MC3_18")
    print("=" * 80)

    for lr in lr_list:
        model = BestFusionNet(bert_path).to(DEVICE)
        start_t = time.time()

        val_auc, best_wts = train_eval_single_fold(
            model, train_loader, val_loader, class_wts, lr,
            epochs=25, patience=6, accumulation_steps=8
        )

        print(f"  [-] LR: {lr:<7} | 验证集最佳 AUC: {val_auc:>6.2f}% (总耗时: {time.time() - start_t:>2.0f}s)")

        temp_logs.append({
            "Paradigm": "Best Single-Modal Combine", "Text_Encoder": "TextCNN", "Video_Encoder": "MC3_18",
            "Learning_Rate": lr, "Val_AUC": val_auc, "Weights_Dict": copy.deepcopy(best_wts)
        })
        del model;
        torch.cuda.empty_cache();
        gc.collect()

    best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
    print(f"\n[!] 最优学习率定型 (最佳LR: {best_log['Learning_Rate']})，即将执行测试集盲测...")

    final_model = BestFusionNet(bert_path).to(DEVICE)
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
            "Paradigm": log["Paradigm"], "Text_Encoder": log["Text_Encoder"], "Video_Encoder": log["Video_Encoder"],
            "Learning_Rate": log["Learning_Rate"], "Val_AUC": log["Val_AUC"],
            "Test_UAR": log.get("Test_UAR") if is_op else None,
            "Test_WAR": log.get("Test_WAR") if is_op else None,
            "Test_AUC": log.get("Test_AUC") if is_op else None,
            "Is_Optimal": is_op
        })

    df_results = pd.DataFrame(all_results)
    df_optimal = df_results[df_results["Is_Optimal"] == True]

    # 【核心修改】: 补充 LaTeX 表格自动生成逻辑
    latex_str = "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
    latex_str += "\\textbf{Paradigm} & \\textbf{Text + Video Encoders} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\midrule\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Paradigm']} & {row['Text_Encoder']} + {row['Video_Encoder']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Performance of the Best Single-Modality Fusion (TextCNN + MC3\\_18)}\n\\label{tab:best_fusion_ablation}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    if len(os.listdir(DATASET_BASE_DIR)) > 0:
        tr_recs, val_recs, ts_recs, class_wts = load_aligned_video_text_dataset_from_csv(CSV_PATH, DATASET_BASE_DIR)

        if len(tr_recs) > 0:
            df_results, latex_snippet = run_best_model_combination(tr_recs, val_recs, ts_recs, class_wts, BERT_PATH)

            print("\n--- 供论文引用的 最优模态组合结果表 ---")
            print(latex_snippet)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

            # 【核心修改】: 同时保存 CSV 结果和 LaTeX 代码片段文件
            output_csv_path = os.path.join(OUTPUT_BASE_DIR, f"Best_Model_Fusion_{current_time}.csv")
            output_txt_path = os.path.join(OUTPUT_BASE_DIR, f"latex_table_best_fusion_{current_time}.txt")

            df_results.to_csv(output_csv_path, index=False)
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(latex_snippet)

            print(f"\n 实验完成！CSV及LaTeX代码已分别保存至:")
            print(f"  - {output_csv_path}")
            print(f"  - {output_txt_path}")