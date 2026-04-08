import os
import time
import datetime
import argparse
import pandas as pd
import numpy as np
import warnings
import logging

# ==========================================
# 0. Environment Setup & Logging Suppression
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jieba
import jieba.posseg as pseg
logging.getLogger('jieba').setLevel(logging.ERROR)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import torch
from transformers import BertTokenizer, BertModel


# ==========================================
# Module 1: Data Loading & Parsing
# ==========================================
def load_real_text_data(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    raw_texts = []
    labels = []
    groups = []

    print(f"[Data Loading] Scanning text directory: {base_dir}")

    for label_str, label_idx in label_map.items():
        label_dir = os.path.join(base_dir, label_str)
        if not os.path.exists(label_dir): continue

        for subject_folder in os.listdir(label_dir):
            subject_dir = os.path.join(label_dir, subject_folder)
            if not os.path.isdir(subject_dir): continue

            for task_folder in os.listdir(subject_dir):
                task_dir = os.path.join(subject_dir, task_folder)
                if not os.path.isdir(task_dir): continue

                for file in os.listdir(task_dir):
                    if file.endswith('.txt'):
                        file_path = os.path.join(task_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                        except UnicodeDecodeError:
                            try:
                                with open(file_path, 'r', encoding='gbk') as f:
                                    text = f.read().strip()
                            except Exception:
                                continue

                        if text:
                            raw_texts.append(text)
                            labels.append(label_idx)
                            groups.append(subject_folder)

    print(f"[Data Loading] Complete! Loaded {len(raw_texts)} valid text samples.")
    return raw_texts, np.array(labels), np.array(groups)


# ==========================================
# Module 2: Feature Extraction (Linguistic + BERT)
# ==========================================
def extract_linguistic_features(text):
    if not text or len(text.strip()) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    sentences = text.replace('?', '。').replace('!', '。').replace('；', '。').split('。')
    sentences = [s for s in sentences if len(s.strip()) > 0]
    num_sentences = len(sentences) if len(sentences) > 0 else 1

    avg_sentence_length = len(text) / num_sentences

    words = pseg.cut(text)
    word_list = []
    pronoun_count = 0
    noun_count = 0

    for word, flag in words:
        word_list.append(word)
        if flag.startswith('r'): pronoun_count += 1
        elif flag.startswith('n'): noun_count += 1

    total_words = len(word_list) if len(word_list) > 0 else 1

    pronoun_ratio = pronoun_count / total_words
    noun_ratio = noun_count / total_words
    ttr = len(set(word_list)) / total_words

    return [avg_sentence_length, pronoun_ratio, noun_ratio, ttr]


def extract_bert_features(text_list, model_name='bert-base-chinese'):
    print(f"[Feature Extraction] Loading BERT model and tokenizer ({model_name})...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"[Error] Failed to load BERT model. Check path or internet connection: {e}")
        return None

    features = []
    print(f"[Feature Extraction] Extracting BERT semantic features for {len(text_list)} texts...")

    with torch.no_grad():
        for i, text in enumerate(text_list):
            if not text or len(text.strip()) == 0:
                features.append(np.zeros(768))
                continue

            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            features.append(cls_embedding)

            if (i + 1) % 200 == 0 or (i + 1) == len(text_list):
                print(f"  - Processed {i + 1}/{len(text_list)} texts")

    return np.array(features)


# ==========================================
# Module 3: Baseline Experiments
# ==========================================
def run_baseline_experiments(X, y, groups):
    print("\n[Model Evaluation] Executing Official Single Fixed Split...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    test_subjects = np.unique(groups[test_idx])
    print(f"  -> Split Summary: Train samples: {len(train_idx)}, Test samples: {len(test_idx)} (from {len(test_subjects)} subjects)")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "MLP (Deep Learning)": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    }

    results = []
    print(f"\n{'Model':<25} | {'UAR (%)':<10} | {'WAR (%)':<10} | {'AUC (%)':<10}")
    print("-" * 65)

    for name, model in models.items():
        start_t = time.time()
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

            uar = recall_score(y_test, y_pred, average='macro') * 100
            war = recall_score(y_test, y_pred, average='weighted') * 100
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro') * 100
            except ValueError:
                auc = np.nan

            results.append({"Model": name, "UAR": uar, "WAR": war, "AUC": auc})
            print(f"{name:<25} | {uar:>7.2f}    | {war:>7.2f}    | {auc:>7.2f}    (Time: {time.time()-start_t:>2.0f}s)")
        except Exception as e:
            print(f"{name:<25} | Failed: {str(e)}")

    df_results = pd.DataFrame(results)

    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Text Modality Machine Learning Baseline (Official 80/20 Fixed Split)}\n\\label{tab:ml_text_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# Module 4: Output Persistence
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"Text_ML_Benchmark_{current_time}")

    try:
        os.makedirs(full_save_path, exist_ok=True)
        print(f"\n[Save] Created results directory: {full_save_path}")

        df_results.to_csv(os.path.join(full_save_path, "benchmark_metrics.csv"), index=False)
        pd.DataFrame({'Subject_ID': test_subjects}).to_csv(os.path.join(full_save_path, "official_test_subjects.csv"), index=False)

        with open(os.path.join(full_save_path, "latex_table_code.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_str)

        print(f"[Save] Benchmark results saved successfully.")
    except Exception as e:
        print(f"[Error] Failed to save outputs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Modality Machine Learning Baseline")
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the text dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_text_ML', help='Directory to save results')
    parser.add_argument('--bert_path', type=str, default='bert-base-chinese', help='Path to local BERT model or HF model name')
    args = parser.parse_args()

    raw_texts, labels, groups = load_real_text_data(args.data_dir)

    if len(raw_texts) == 0:
        print("[Terminated] No valid text data loaded.")
    else:
        print("\n[Phase 1] Extracting handcrafted linguistic features...")
        linguistic_features = np.array([extract_linguistic_features(text) for text in raw_texts])

        print("\n[Phase 2] Extracting BERT deep semantic features...")
        bert_features = extract_bert_features(raw_texts, args.bert_path)

        if bert_features is not None:
            print("\n[Feature Fusion] Concatenating linguistic and BERT features...")
            X_fused = np.hstack((bert_features, linguistic_features))

            print("\n[Phase 3] Running baseline model validation...")
            df_results, latex_snippet, test_subjects = run_baseline_experiments(X_fused, labels, groups)

            save_outputs(df_results, latex_snippet, test_subjects, args.output_dir)
        else:
            print("\n[Terminated] Program stopped due to BERT extraction failure.")