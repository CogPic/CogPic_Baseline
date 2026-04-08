import os
import time
import datetime
import copy
import argparse
import pandas as pd
import numpy as np
import warnings
import gc
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

# ==========================================
# 0. Environment Setup
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Uncomment for mainland China

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models.video as video_models
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Module 1: Model Definitions
# ==========================================
class Video3DCNNWrapper(nn.Module):
    def __init__(self, model_name='r3d_18', num_classes=3):
        super().__init__()
        if model_name == 'r3d_18':
            self.model = video_models.r3d_18(pretrained=True)
        elif model_name == 'mc3_18':
            self.model = video_models.mc3_18(pretrained=True)
        elif model_name == 'r2plus1d_18':
            self.model = video_models.r2plus1d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=3, freeze_cnn=True, weight_path=None):
        super().__init__()
        print(" -> [Weights Loading] ResNet18 -> LSTM ...")

        # Load local weights if provided, else download via timm
        if weight_path and os.path.exists(weight_path):
            backbone = timm.create_model('resnet18', pretrained=False, num_classes=1000, checkpoint_path=weight_path)
        else:
            backbone = timm.create_model('resnet18', pretrained=True, num_classes=1000)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)  # 512 = 256*2

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)
        out, _ = self.lstm(features)
        return self.fc(out.mean(dim=1))


class ResNetTransformer(nn.Module):
    def __init__(self, num_classes=3, freeze_cnn=True, weight_path=None):
        super().__init__()
        print(" -> [Weights Loading] ResNet18 -> Transformer ...")

        if weight_path and os.path.exists(weight_path):
            backbone = timm.create_model('resnet18', pretrained=False, num_classes=1000, checkpoint_path=weight_path)
        else:
            backbone = timm.create_model('resnet18', pretrained=True, num_classes=1000)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)
        return self.fc(self.transformer(features).mean(dim=1))


def build_video_model(model_name, num_classes=3, resnet_weight_path=None):
    if model_name == 'R3D_18':
        return Video3DCNNWrapper('r3d_18', num_classes)
    elif model_name == 'MC3_18':
        return Video3DCNNWrapper('mc3_18', num_classes)
    elif model_name == 'R2Plus1D':
        return Video3DCNNWrapper('r2plus1d_18', num_classes)
    elif model_name == 'ResNet+LSTM':
        return ResNetLSTM(num_classes, weight_path=resnet_weight_path)
    elif model_name == 'ResNet+Transformer':
        return ResNetTransformer(num_classes, weight_path=resnet_weight_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==========================================
# Module 2: Dataset Pipeline
# ==========================================
def load_video_dataset_from_csv(csv_path, base_dir):
    print(f"\n[Data Loading] Reading master split: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if '120' in d:
                task_id = os.path.basename(root)
                path_mapping[task_id] = os.path.join(root, d)

    train_records, val_records, test_records = [], [], []
    train_labels = []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            record = {'folder_path': path_mapping[task_id], 'label': label}
            if split_type == 'Train':
                train_records.append(record)
                train_labels.append(label)
            elif split_type == 'Validation':
                val_records.append(record)
            elif split_type == 'Test':
                test_records.append(record)

    print(f"[Data Loading] Done! Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"[Class Weights] HC/MCI/AD: {class_weights_tensor.cpu().numpy()}")

    return train_records, val_records, test_records, class_weights_tensor


class VideoFramesDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pt_path = os.path.join(rec['folder_path'], "video_tensor.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location='cpu', weights_only=True), rec['label']
        return torch.zeros(3, 120, 224, 224), rec['label']


# ==========================================
# Module 3: Training Function (With Alignment Fix)
# ==========================================
def train_with_val(model, train_loader, val_loader, epochs, lr, weight_decay, patience, class_weights_tensor):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler(device=DEVICE.type)

    best_val_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"Training Started | LR={lr} | Total Epochs: {epochs}")

    for epoch in tqdm(range(epochs), desc="Epoch Progress", leave=True):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        start_epoch_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Batch]", leave=False, ncols=100)):
            batch_start = time.time()

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            # 1. AMP Forward Pass
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 2. Scale Loss and Backward
            scaler.scale(loss).backward()

            # 3. CRITICAL ALIGNMENT FIX: Unscale and apply gradient clipping (max_norm=5.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)

            # 4. Step and Update
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - batch_start
            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                avg_loss = epoch_loss / batch_count
                print(f"  -> Epoch {epoch + 1:2d} | Batch {batch_idx + 1:3d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | Batch Time: {batch_time:.3f}s")

        # Validation Phase
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type):
                    logits = model(inputs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(targets.numpy())

        try:
            val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro') * 100
        except ValueError:
            val_auc = 0.0

        epoch_time = time.time() - start_epoch_time
        print(f"  [Epoch {epoch + 1:2d}/{epochs}] Val AUC: {val_auc:.2f}% | Epoch Time: {epoch_time:.1f}s\n")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  [Early Stopping] No improvement for {patience} consecutive epochs.")
            break

    model.load_state_dict(best_model_wts)
    return best_val_auc, best_model_wts


def evaluate_on_test(model, test_loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type):
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


# ==========================================
# Module 4: Main Execution
# ==========================================
def run_video_dl_experiments(train_records, val_records, test_records, class_weights_tensor, resnet_weight_path):
    print(f"\n[Environment] Device: {DEVICE.type.upper()}")

    train_dataset = VideoFramesDataset(train_records)
    val_dataset = VideoFramesDataset(val_records)
    test_dataset = VideoFramesDataset(test_records)

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True)

    model_names = ["R3D_18", "MC3_18", "R2Plus1D", "ResNet+LSTM", "ResNet+Transformer"]
    lr_list = [1e-3, 5e-4, 1e-4, 5e-5]
    MAX_EPOCHS, PATIENCE, WEIGHT_DECAY = 50, 7, 1e-4

    all_detailed_results = []

    print("\n" + "=" * 90)
    print("Starting Comprehensive Video Model Tuning (Aligned Gradients)")
    print("=" * 90)

    for name in model_names:
        print(f"\n>>> Tuning Model: {name} <<<")
        temp_logs = []

        for lr in lr_list:
            model = build_video_model(name, 3, resnet_weight_path).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_with_val(
                model, train_loader, val_loader, MAX_EPOCHS, lr, WEIGHT_DECAY, PATIENCE, class_weights_tensor
            )

            print(f"  [LR={lr}] Best Val AUC: {val_auc:.2f}% (Time {time.time() - start_t:.0f}s)")

            temp_logs.append({
                "Model": name,
                "Learning_Rate": lr,
                "Val_AUC": val_auc,
                "Weights_Dict": copy.deepcopy(best_wts)
            })

            del model
            torch.cuda.empty_cache()
            gc.collect()

        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(f"[Optimal Config] {name} -> Best LR: {best_log['Learning_Rate']}, Val AUC: {best_log['Val_AUC']:.2f}%")

        final_model = build_video_model(name, 3, resnet_weight_path).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])
        test_uar, test_war, test_auc = evaluate_on_test(final_model, test_loader)
        print(f"[Final Test] AUC: {test_auc:.2f}% | UAR: {test_uar:.2f}% | WAR: {test_war:.2f}%\n")

        del final_model
        gc.collect()

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

    print("\n" + "=" * 70)
    print("Final Optimal Configurations Performance")
    print("=" * 70)
    print(f"{'Model':<20} | {'Best LR':<10} | {'Test UAR (%)':<12} | {'Test WAR (%)':<12} | {'Test AUC (%)':<12}")
    print("-" * 80)
    for _, row in df_optimal.iterrows():
        print(
            f"{row['Model']:<20} | {row['Learning_Rate']:<10.0e} | {row['Test_UAR']:>12.2f} | {row['Test_WAR']:>12.2f} | {row['Test_AUC']:>12.2f}")

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Model']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Video Deep Models Performance}\n\\label{tab:video_dl_performance}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Video Deep Learning Pipeline")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for video dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_video_DL', help='Directory to save outputs')
    parser.add_argument('--resnet_weight', type=str, default=None,
                        help='Path to local ResNet18 timm weights (optional)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_recs, val_recs, test_recs, class_wts = load_video_dataset_from_csv(args.csv_path, args.data_dir)

    if len(train_recs) > 0 and len(test_recs) > 0 and len(val_recs) > 0:
        df_results, latex_snippet = run_video_dl_experiments(train_recs, val_recs, test_recs, class_wts,
                                                             args.resnet_weight)

        print("\n--- LaTeX Table Snippet for Citation ---")
        print(latex_snippet)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(args.output_dir, f"Video_Opt_Aligned_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        df_results.to_csv(os.path.join(full_save_path, "detailed_grid_search_metrics.csv"), index=False)
        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[Execution Complete] Results saved to: {full_save_path}")
    else:
        print("[Error] Data loading failed. Please verify the paths.")