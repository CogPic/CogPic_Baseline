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

# Optional: Uncomment the following line to speed up HuggingFace downloads in mainland China
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import timm
import soundfile as sf
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Module 1: Deep Learning Model Library
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


def build_model(model_name, num_classes=3, weight_path=None):
    if model_name == "ResNet18":
        model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    elif model_name == "ResNetSE":
        model = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000)
        # Load local pretrained weights if provided
        if weight_path and os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                f" [Warning] Pretrained weight path for {model_name} not found or not provided. Training from scratch.")
        model.reset_classifier(num_classes)
    elif model_name == "CRNN":
        model = CustomCRNN(num_classes=num_classes)
    elif model_name == "ViT (AST Surrogate)":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


# ==========================================
# Module 2: Audio Processing Pipeline (Spectrogram)
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

        # Align with ImageNet mean and variance for pretrained visual backbones
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

            # Instance-level standardization (Eliminate global volume differences)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

            # Convert to pseudo-color 3 channels
            mel_rgb = mel_db.repeat(3, 1, 1)

            # Apply ImageNet normalization
            mel_rgb = self.imagenet_normalize(mel_rgb)

            mel_final = F.interpolate(mel_rgb.unsqueeze(0), size=(224, 224), mode='bilinear',
                                      align_corners=False).squeeze(0)
            return mel_final, self.labels[idx]

        except Exception as e:
            print(f" [Warning] Failed to read audio {file_path}: {e}")
            return torch.zeros((3, 224, 224)), self.labels[idx]


def load_audio_dataset_from_csv(csv_path, audio_base_dir):
    print(f"\n[Data Loading] Reading master split list: {csv_path}")
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

    print(f"[Data Loading] Done! Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Automatically calculate class weights to solve the imbalance problem
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"[Data Distribution] Training set loss weights (HC/MCI/AD): {class_weights_tensor.cpu().numpy()}")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_weights_tensor


# ==========================================
# Module 3: Model Training and Evaluation
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
                             class_weights_tensor, weight_path=None):
    print(f"\n[Environment] Compute Device: {DEVICE.type.upper()}")

    train_ds = FixedAudioDataset(train_paths, train_labels)
    val_ds = FixedAudioDataset(val_paths, val_labels)
    test_ds = FixedAudioDataset(test_paths, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model_names = ["ResNet18", "ResNetSE", "CRNN", "ViT (AST Surrogate)"]

    # Separate hyperparameter pools to protect giant models
    lr_list_cnn = [1e-3, 5e-4, 1e-4, 5e-5]
    lr_list_vit = [5e-5, 3e-5, 2e-5, 1e-5]
    WEIGHT_DECAY = 1e-4

    all_detailed_results = []

    print("\n" + "=" * 70)
    print("Starting Grid Search (Best model evaluated on blind test set)")
    print("=" * 70)

    for name in model_names:
        print(f"\n>>> Tuning Model: {name} <<<")
        is_vit = ("ViT" in name)
        current_lr_list = lr_list_vit if is_vit else lr_list_cnn

        # Dynamic convergence strategy
        MAX_EPOCHS = 10 if is_vit else 50
        PATIENCE = 3 if is_vit else 7

        temp_logs = []

        # 1. Grid search on the validation set
        for current_lr in current_lr_list:
            model = build_model(name, num_classes=3, weight_path=weight_path).to(DEVICE)
            start_t = time.time()

            val_auc, best_wts = train_eval_single_fold(
                model, train_loader, val_loader,
                epochs=MAX_EPOCHS, lr=current_lr, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, class_weights_tensor=class_weights_tensor
            )

            print(f"  [-] LR: {current_lr:<7} | Val AUC: {val_auc:>6.2f}% (Time: {time.time() - start_t:>2.0f}s)")

            temp_logs.append({
                "Model": name,
                "Learning_Rate": current_lr,
                "Val_AUC": val_auc,
                "Weights_Dict": copy.deepcopy(best_wts)
            })

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        # 2. Lock optimal hyperparameters
        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(
            f"[!] {name} Tuning Complete -> Best LR: {best_log['Learning_Rate']} (Best Val AUC: {best_log['Val_AUC']:.2f}%)")

        # 3. Final Test (Single forward pass on the Test Set)
        print(f"[*] Unlocking blind test set for final evaluation...")
        final_model = build_model(name, num_classes=3, weight_path=weight_path).to(DEVICE)
        final_model.load_state_dict(best_log['Weights_Dict'])

        test_uar, test_war, test_auc = evaluate_model(final_model, test_loader)
        print(f"    --> Final Test Performance: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%\n")

        del final_model
        gc.collect()

        # 4. Summarize results
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

    # Generate detailed DataFrame
    df_results = pd.DataFrame(all_detailed_results)

    # Generate LaTeX code for paper citation (Optimal results only)
    df_optimal = df_results[df_results["Is_Optimal_Config"] == True]

    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{Best LR} & \\textbf{Test UAR (\\%)} & \\textbf{Test WAR (\\%)} & \\textbf{Test AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_optimal.iterrows():
        latex_str += f"{row['Model']} & {row['Learning_Rate']:.0e} & {row['Test_UAR']:.2f} & {row['Test_WAR']:.2f} & {row['Test_AUC']:.2f} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Audio Deep Models Performance on Independent Test Set}\n\\label{tab:audio_dl_aligned}\n\\end{table}"

    return df_results, latex_str


# ==========================================
# Module 4: Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Modality Deep Learning Pipeline (Spectrogram-based)")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the audio dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_audio_DL', help='Directory to save outputs')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to pre-trained SEResNet50 weights')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.weight_path and os.path.exists(args.weight_path):
        print("Testing SEResNet50 weight loading...")
        test_model = timm.create_model('seresnet50.ra2_in1k', pretrained=False, num_classes=1000)
        state_dict = torch.load(args.weight_path, map_location='cpu')
        test_model.load_state_dict(state_dict, strict=True)
        test_model.reset_classifier(3)
        print("SEResNet50 weights loaded successfully!\n")
    elif args.weight_path:
        print(f"[Warning] Provided weight path does not exist: {args.weight_path}\n")

    tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts = load_audio_dataset_from_csv(args.csv_path, args.data_dir)

    if len(tr_p) > 0 and len(ts_p) > 0 and len(val_p) > 0:
        df_results, latex_snippet = run_audio_dl_experiments(
            tr_p, tr_y, val_p, val_y, ts_p, ts_y, class_wts, args.weight_path
        )

        print("\n--- LaTeX Table Snippet for Paper (Optimal Performance Only) ---")
        print(latex_snippet)
        print("----------------------------------------------------------------\n")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(args.output_dir, f"Audio_Opt_Aligned_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        csv_save_path = os.path.join(full_save_path, "detailed_grid_search_metrics.csv")
        df_results.to_csv(csv_save_path, index=False)

        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[Execution Complete] Optimization results saved to: {full_save_path}")
        print(f" -> Detailed hyperparameter search logs saved as: detailed_grid_search_metrics.csv")
    else:
        print(
            "[Terminated] Dataset extraction failed. Please check the CSV file content or the audio source directory.")