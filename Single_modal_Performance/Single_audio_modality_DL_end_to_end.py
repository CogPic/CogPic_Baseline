import os
import time
import datetime
import copy
import argparse
import pandas as pd
import numpy as np
import warnings
import gc

# Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, roc_auc_score

# ==========================================
# 0. Environment Setup
# ==========================================
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
# Module 1: End-to-End Raw Waveform Deep Learning Model
# ==========================================
class RawWaveLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(RawWaveLSTM, self).__init__()

        # 1. Front-end 1D-CNN Downsampling Module
        # Input shape: (Batch, 1, 80000)
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
        # Output shape after frontend: (Batch, 256, 156)

        # 2. Temporal Modeling LSTM
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )

        # 3. Classification Head
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
# Module 2: Raw Waveform Processing Pipeline
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

            # Convert to mono if multi-channel
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            elif waveform.ndim == 0:
                waveform = waveform.unsqueeze(0)

            # Resample if necessary
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

            # Truncate or pad to target length
            num_samples = waveform.shape[0]
            if num_samples > self.target_samples:
                waveform = waveform[:self.target_samples]
            elif num_samples < self.target_samples:
                padding = self.target_samples - num_samples
                waveform = F.pad(waveform, (0, padding))

            # Z-score normalization to eliminate absolute volume differences
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
            waveform = waveform.unsqueeze(0)  # Shape: (1, 80000)

            return waveform, self.labels[idx]
        except Exception as e:
            print(f" [Warning] Failed to read audio {file_path}: {e}")
            return torch.zeros((1, self.target_samples)), self.labels[idx]


def load_audio_dataset_from_csv(csv_path, audio_base_dir):
    print(f"\n[Data Loading] Reading master split list: {csv_path}")
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

    print(f"[Data Loading] Done! Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Automatically calculate class weights to solve class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_weights_tensor


# ==========================================
# Module 3: Rigorous Blind Test Logging and Tuning (Unified FP32)
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)

            # Pure FP32 inference
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

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Pure FP32 forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Standard FP32 backward pass with gradient clipping (essential for LSTM)
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
    print(f"\n[Environment] Compute Device: {DEVICE.type.upper()}")

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
    print("Starting End-to-End Raw Waveform Model (RawWave-LSTM) Tuning (Unified FP32)")
    print("=" * 70)

    for current_lr in lr_list:
        model = RawWaveLSTM(num_classes=3).to(DEVICE)
        start_t = time.time()

        val_auc, best_wts = train_eval_single_fold(
            model, train_loader, val_loader,
            epochs=MAX_EPOCHS, lr=current_lr, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, class_weights_tensor=class_wts
        )

        print(f"  [-] LR: {current_lr:<7} | Val AUC: {val_auc:>6.2f}% (Time: {time.time() - start_t:>2.0f}s)")

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
    print(f"\n[!] Tuning Complete -> Best LR: {best_log['Learning_Rate']} (Best Val AUC: {best_log['Val_AUC']:.2f}%)")

    print(f"[*] Unlocking blind test set for final evaluation...")
    final_model = RawWaveLSTM(num_classes=3).to(DEVICE)
    final_model.load_state_dict(best_log['Weights_Dict'])

    test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
    print(f"    --> Final Test Performance: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%\n")

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
# Module 4: Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Waveform Deep Learning Pipeline (1D-CNN + LSTM)")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the raw audio dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/RawWave_LSTM_Experiment',
                        help='Directory to save outputs')

    args = parser.parse_args()

    # Create output directory if it doesn't exist to prevent IO errors
    os.makedirs(args.output_dir, exist_ok=True)

    tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts = load_audio_dataset_from_csv(args.csv_path, args.data_dir)

    if len(tr_p) > 0 and len(ts_p) > 0 and len(val_p) > 0:
        df_results, latex_snippet = run_raw_wave_experiment(tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts)

        print("\n--- LaTeX Table Snippet for Paper (Optimal Performance Only) ---")
        print(latex_snippet)
        print("----------------------------------------------------------------\n")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(args.output_dir, f"RawWave_Opt_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        df_results.to_csv(os.path.join(full_save_path, "detailed_grid_search_metrics.csv"), index=False)
        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[Execution Complete] Optimization results saved to: {full_save_path}")
        print(f" -> Detailed hyperparameter search logs saved as: detailed_grid_search_metrics.csv")
    else:
        print(
            "[Terminated] Dataset extraction failed. Please check the CSV file content or the audio source directory.")