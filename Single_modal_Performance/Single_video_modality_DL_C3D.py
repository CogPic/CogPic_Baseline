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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. C3D Model Definition
# ==========================================
class C3D(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc6 = nn.Linear(512, 2048)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        return self.fc8(x)


# ==========================================
# 2. Dataset Module
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

    train_records, val_records, test_records, train_labels = [], [], [], []

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
    return train_records, val_records, test_records, class_weights_tensor


class VideoFramesDataset(Dataset):
    def __init__(self, records): self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pt_path = os.path.join(rec['folder_path'], "video_tensor.pt")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location='cpu', weights_only=True), rec['label']
        return torch.zeros(3, 120, 224, 224), rec['label']


# ==========================================
# 3. Training and Evaluation Function
# ==========================================
def train_with_val(model, train_loader, val_loader, epochs, lr, weight_decay, patience, class_weights_tensor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler(device=DEVICE.type)

    best_val_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs), desc="Epoch Progress", leave=True):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Batch]", leave=False)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            # 1. AMP Forward Pass
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 2. Scale Loss and Backward
            scaler.scale(loss).backward()

            # 3. Unscale and Clip Gradients (Crucial for preventing C3D explosion)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 4. Step and Update
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

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

        print(
            f"  [Epoch {epoch + 1:2d}/{epochs}] Val AUC: {val_auc:.2f}% | Avg Loss: {epoch_loss / len(train_loader):.4f}")

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
# 4. Main Execution
# ==========================================
def run_c3d_experiment_with_outputs(train_records, val_records, test_records, class_weights_tensor):
    train_dataset = VideoFramesDataset(train_records)
    val_dataset = VideoFramesDataset(val_records)
    test_dataset = VideoFramesDataset(test_records)

    # Note: Reduce batch_size if OOM occurs
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6, pin_memory=True)

    lr_list = [1e-3, 5e-4, 1e-4, 5e-5]
    MAX_EPOCHS, PATIENCE, WEIGHT_DECAY = 50, 7, 1e-4
    all_detailed_results = []

    print("\n" + "=" * 80)
    print("Starting Independent C3D Experiment (AMP + Gradient Clipping)")
    print("=" * 80)

    temp_logs = []

    for lr in lr_list:
        model = C3D(num_classes=3).to(DEVICE)
        val_auc, best_wts = train_with_val(
            model, train_loader, val_loader, MAX_EPOCHS, lr, WEIGHT_DECAY, PATIENCE, class_weights_tensor
        )
        print(f"  [LR={lr}] Best Val AUC: {val_auc:.2f}%")
        temp_logs.append({
            "Model": "C3D",
            "Learning_Rate": lr,
            "Val_AUC": val_auc,
            "Weights_Dict": copy.deepcopy(best_wts)
        })
        del model
        torch.cuda.empty_cache()
        gc.collect()

    best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
    print(f"\n[Optimal Config] C3D -> Best LR: {best_log['Learning_Rate']}, Val AUC: {best_log['Val_AUC']:.2f}%")

    final_model = C3D(num_classes=3).to(DEVICE)
    final_model.load_state_dict(best_log['Weights_Dict'])
    test_uar, test_war, test_auc = evaluate_on_test(final_model, test_loader)

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
    latex_str += "\\hline\n\\end{tabular}\n\\caption{C3D Model Performance}\n\\label{tab:c3d_performance}\n\\end{table}"

    return df_results, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C3D Video Deep Learning Pipeline")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for video dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_video_DL', help='Directory to save outputs')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_recs, val_recs, test_recs, class_wts = load_video_dataset_from_csv(args.csv_path, args.data_dir)

    if train_recs and test_recs and val_recs:
        df_results, latex_snippet = run_c3d_experiment_with_outputs(train_recs, val_recs, test_recs, class_wts)

        print("\n--- LaTeX Table Snippet for Citation ---")
        print(latex_snippet)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(args.output_dir, f"C3D_Only_Results_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        csv_save_path = os.path.join(full_save_path, "c3d_grid_search_metrics.csv")
        df_results.to_csv(csv_save_path, index=False)
        with open(os.path.join(full_save_path, "c3d_latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[Execution Complete] Results and tables exported to: {full_save_path}")
    else:
        print("[Error] Data loading failed. Please verify the paths.")