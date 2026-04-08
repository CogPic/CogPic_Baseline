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
# 0. Environment & Logging Setup
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Uncomment for mainland China users

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.optim import AdamW
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Module 1: Deep Learning Model Definitions
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=3, num_filters=100, filter_sizes=(3, 4, 5)):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_classes=3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled_out = torch.mean(out, dim=1)
        return self.fc(pooled_out)


class AttentionBiLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_classes=3):
        super(AttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention_weights(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)


# ==========================================
# Module 2: Data Loading & Precomputation
# ==========================================
def load_text_dataset_from_csv(csv_path, base_dir, model_path, max_len=256):
    print(f"\n[Data Loading] Reading master split list: {csv_path}")
    df = pd.read_csv(csv_path)

    path_mapping = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                task_id = os.path.basename(root)
                path_mapping[task_id] = os.path.join(root, file)

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    for _, row in df.iterrows():
        task_id = str(row['Task_ID']).strip()
        label_idx = int(row['Label_Idx'])
        split_type = str(row['Split']).strip()

        if task_id in path_mapping:
            file_path = path_mapping[task_id]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except:
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        text = f.read().strip()
                except:
                    text = ""

            if text:
                if split_type == 'Train':
                    train_texts.append(text)
                    train_labels.append(label_idx)
                elif split_type == 'Validation':
                    val_texts.append(text)
                    val_labels.append(label_idx)
                elif split_type == 'Test':
                    test_texts.append(text)
                    test_labels.append(label_idx)

    print(f"[Data Loading] Done! Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Calculate class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"[Data Distribution] Automatically computed class weights (HC/MCI/AD): {class_weights_tensor.cpu().numpy()}")

    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Encode splits
    train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    val_enc = tokenizer(val_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    train_ds = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(train_labels))
    val_ds = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(val_labels))
    test_ds = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], torch.tensor(test_labels))

    return train_ds, val_ds, test_ds, class_weights_tensor


def precompute_embeddings(dataset, static_bert, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embs, all_labels = [], []
    with torch.no_grad():
        for input_ids, masks, labels in loader:
            input_ids, masks = input_ids.to(DEVICE), masks.to(DEVICE)
            outputs = static_bert(input_ids, attention_mask=masks)
            all_embs.append(outputs.last_hidden_state.cpu())
            all_labels.append(labels.cpu())
    return TensorDataset(torch.cat(all_embs), torch.cat(all_labels))


# ==========================================
# Module 3: Training and Evaluation
# ==========================================
def evaluate_model(model, dataloader, is_finetune):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            if is_finetune:
                input_ids, masks, labels = [b.to(DEVICE) for b in batch]
                logits = model(input_ids, attention_mask=masks).logits
            else:
                embeddings, labels = [b.to(DEVICE) for b in batch]
                logits = model(embeddings)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_labels.extend(labels.cpu().numpy())

    uar = recall_score(all_labels, all_preds, average='macro') * 100
    war = recall_score(all_labels, all_preds, average='weighted') * 100
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except ValueError:
        auc = 0.0
    return uar, war, auc


def train_eval_single_fold(model, train_loader, val_loader, is_finetune, epochs, lr, weight_decay, patience,
                           class_weights_tensor):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Inject class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if is_finetune:
                input_ids, masks, labels = [b.to(DEVICE) for b in batch]
                # HuggingFace default loss doesn't use class weights, so we compute it manually
                logits = model(input_ids, attention_mask=masks).logits
                loss = criterion(logits, labels)
            else:
                embeddings, labels = [b.to(DEVICE) for b in batch]
                outputs = model(embeddings)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        _, _, val_auc = evaluate_model(model, val_loader, is_finetune)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_auc, best_model_wts


def run_text_dl_experiments(train_ds, val_ds, test_ds, class_weights_tensor, model_path):
    print(f"\n[Environment] Compute Device: {DEVICE.type.upper()}")

    # Load static BERT for feature precomputation
    static_bert = BertModel.from_pretrained(model_path).to(DEVICE)
    static_bert.eval()

    print("\n[Feature Engineering] Precomputing static BERT features, please wait...")
    train_emb_ds = precompute_embeddings(train_ds, static_bert)
    val_emb_ds = precompute_embeddings(val_ds, static_bert)
    test_emb_ds = precompute_embeddings(test_ds, static_bert)

    del static_bert
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Build DataLoaders
    train_loader_ft = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader_ft = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader_ft = DataLoader(test_ds, batch_size=16, shuffle=False)

    train_loader_st = DataLoader(train_emb_ds, batch_size=16, shuffle=True)
    val_loader_st = DataLoader(val_emb_ds, batch_size=16, shuffle=False)
    test_loader_st = DataLoader(test_emb_ds, batch_size=16, shuffle=False)

    model_names = ["TextCNN", "BiLSTM", "Attention-BiLSTM", "BERT Fine-tune"]
    lr_list_static = [1e-3, 5e-4, 1e-4, 5e-5]
    lr_list_finetune = [5e-5, 3e-5, 2e-5, 1e-5]

    WEIGHT_DECAY = 1e-4
    all_detailed_results = []

    print("\n" + "=" * 70)
    print("Starting Rigorous Grid Search (Best model evaluated on blind test set)")
    print("=" * 70)

    for name in model_names:
        print(f"\n>>> Tuning Model: {name} <<<")
        is_ft = (name == "BERT Fine-tune")
        current_lr_list = lr_list_finetune if is_ft else lr_list_static

        # Dynamic epochs and patience mapping
        MAX_EPOCHS = 5 if is_ft else 30
        PATIENCE = 3 if is_ft else 7

        cur_train_loader = train_loader_ft if is_ft else train_loader_st
        cur_val_loader = val_loader_ft if is_ft else val_loader_st
        cur_test_loader = test_loader_ft if is_ft else test_loader_st

        temp_logs = []

        # 1. Grid search for optimal learning rate on validation set
        for current_lr in current_lr_list:
            if name == "TextCNN":
                model = TextCNN().to(DEVICE)
            elif name == "BiLSTM":
                model = BiLSTMModel().to(DEVICE)
            elif name == "Attention-BiLSTM":
                model = AttentionBiLSTM().to(DEVICE)
            elif name == "BERT Fine-tune":
                model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3).to(DEVICE)

            start_t = time.time()
            val_auc, best_wts = train_eval_single_fold(
                model, cur_train_loader, cur_val_loader,
                is_finetune=is_ft, epochs=MAX_EPOCHS, lr=current_lr,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE, class_weights_tensor=class_weights_tensor
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

        # 2. Extract optimal hyperparameters
        best_log = max(temp_logs, key=lambda x: x["Val_AUC"])
        print(
            f"[!] {name} Tuning Complete -> Best LR: {best_log['Learning_Rate']} (Best Val AUC: {best_log['Val_AUC']:.2f}%)")

        # 3. [Strict Academic Integrity] Only the best config gets evaluated on the blind test set
        print(f"[*] Unlocking blind test set for final evaluation...")
        if name == "TextCNN":
            final_model = TextCNN().to(DEVICE)
        elif name == "BiLSTM":
            final_model = BiLSTMModel().to(DEVICE)
        elif name == "Attention-BiLSTM":
            final_model = AttentionBiLSTM().to(DEVICE)
        elif name == "BERT Fine-tune":
            final_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3).to(DEVICE)

        final_model.load_state_dict(best_log['Weights_Dict'])
        test_uar, test_war, test_auc = evaluate_model(final_model, cur_test_loader, is_ft)
        print(f"    --> Final Test Performance: AUC {test_auc:.2f}% | UAR {test_uar:.2f}% | WAR {test_war:.2f}%\n")

        del final_model
        gc.collect()

        # 4. Summarize tuning logs
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
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Text Deep Models Performance on Independent Test Set}\n\\label{tab:text_dl_fixed}\n\\end{table}"

    return df_results, latex_str


# ==========================================
# Module 4: Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Modality Deep Learning Pipeline")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the text dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_text_DL', help='Directory to save outputs')
    parser.add_argument('--bert_path', type=str, default='bert-base-chinese',
                        help='Path to local BERT weights or HuggingFace model name')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_wts = load_text_dataset_from_csv(args.csv_path, args.data_dir, args.bert_path)

    if len(train_ds) > 0 and len(test_ds) > 0 and len(val_ds) > 0:
        df_results, latex_snippet = run_text_dl_experiments(train_ds, val_ds, test_ds, class_wts, args.bert_path)

        print("\n--- LaTeX Table Snippet for Paper (Optimal Performance Only) ---")
        print(latex_snippet)
        print("----------------------------------------------------------------\n")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = os.path.join(args.output_dir, f"Text_Opt_Aligned_{current_time}")
        os.makedirs(full_save_path, exist_ok=True)

        # Save detailed grid search metrics
        csv_save_path = os.path.join(full_save_path, "detailed_grid_search_metrics.csv")
        df_results.to_csv(csv_save_path, index=False)

        # Save simplified LaTeX table
        with open(os.path.join(full_save_path, "latex_table.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_snippet)

        print(f"\n[Execution Complete] Optimization results saved to: {full_save_path}")
        print(f" -> Detailed hyperparameter search logs saved as: detailed_grid_search_metrics.csv")
    else:
        print("[Terminated] Dataset extraction failed. Please check the CSV file content or the text source directory.")