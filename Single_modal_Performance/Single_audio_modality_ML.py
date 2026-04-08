import os
import time
import datetime
import argparse
import pandas as pd
import numpy as np
import warnings
from collections import Counter

# ==========================================
# 0. Environment Setup
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import opensmile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# ==========================================
# Module 1: Audio Feature Extraction (eGeMAPS)
# ==========================================
def extract_opensmile_features(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    features_list = []
    labels = []
    groups = []

    print(f"\n[Data Loading] Scanning audio directory: {base_dir}")
    print("[Info] Initializing OpenSMILE Engine (Set: eGeMAPSv02, Level: Functionals)...")

    # Initialize OpenSMILE
    # eGeMAPSv02 contains 88 expert acoustic features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    success_count = 0
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
                    if file.endswith('.wav'):
                        file_path = os.path.join(task_dir, file)
                        try:
                            # Extract 88-dimensional features
                            feature_vector = smile.process_file(file_path).values[0]

                            if feature_vector is not None and len(feature_vector) == 88:
                                features_list.append(feature_vector)
                                labels.append(label_idx)
                                groups.append(subject_folder)
                                success_count += 1

                                if success_count % 300 == 0:
                                    print(f"  -> Extracted eGeMAPS features for {success_count} audios...")
                        except Exception as e:
                            print(f"  [Warning] Feature extraction failed for {file_path}: {e}")

    print(f"[Data Loading] Complete! Successfully extracted 88-dim features from {len(features_list)} audios.")
    return np.array(features_list), np.array(labels), np.array(groups)


# ==========================================
# Module 2: Model Building & Stratified Split
# ==========================================
def run_audio_ml_single_split_experiments(X, y, groups):
    print("\n[Model Evaluation] Executing Official 80/20 Stratified Subject Split...")

    unique_subjects = np.unique(groups)
    subject_labels = []

    # Determine class for each subject (majority vote)
    for subj in unique_subjects:
        subj_mask = (groups == subj)
        subj_y = y[subj_mask]
        most_common_label = Counter(subj_y).most_common(1)[0][0]
        subject_labels.append(most_common_label)

    subject_labels = np.array(subject_labels)

    # Subject-level stratified sampling to maintain class balance
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=0.2,
        stratify=subject_labels,
        random_state=42  # Fixed seed for reproducibility
    )

    train_idx = np.isin(groups, train_subjects)
    test_idx = np.isin(groups, test_subjects)

    print(f"  -> Split Summary:")
    print(f"     Total Subjects   : {len(unique_subjects)}")
    print(f"     Train Subjects   : {len(train_subjects)}")
    print(f"     Test Subjects    : {len(test_subjects)}")
    print(f"     Train Samples    : {len(train_idx)}")
    print(f"     Test Samples     : {len(test_idx)}")

    print("\n  === Test Set Subject Distribution ===")
    test_subj_labels = subject_labels[np.isin(unique_subjects, test_subjects)]
    for cls_name, cls_idx in [('HC', 0), ('MCI', 1), ('AD', 2)]:
        count = np.sum(test_subj_labels == cls_idx)
        ratio = count / len(test_subjects) * 100
        print(f"     {cls_name:>3} : {count:3d} subjects ({ratio:5.1f}%)")

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
    print(f"\n{'Audio ML Model':<25} | {'UAR (%)':<10} | {'WAR (%)':<10} | {'AUC (%)':<10}")
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
            print(f"{name:<25} | {uar:>7.2f}    | {war:>7.2f}    | {auc:>7.2f}    (Time: {time.time() - start_t:>2.0f}s)")
        except Exception as e:
            print(f"{name:<25} | Failed: {str(e)}")

    df_results = pd.DataFrame(results)

    # LaTeX Table Generation
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Audio Modality eGeMAPS ML Baseline (Official 80/20 Stratified Subject Split)}\n\\label{tab:ml_audio_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# Module 3: Output Persistence
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"Audio_ML_Benchmark_{current_time}")

    try:
        os.makedirs(full_save_path, exist_ok=True)
        print(f"\n[Save] Created results directory: {full_save_path}")

        df_results.to_csv(os.path.join(full_save_path, "benchmark_metrics.csv"), index=False)
        pd.DataFrame({'Subject_ID': test_subjects}).to_csv(os.path.join(full_save_path, "official_test_subjects.csv"), index=False)

        with open(os.path.join(full_save_path, "latex_table_code.txt"), 'w', encoding='utf-8') as f:
            f.write(latex_str)

        print(f"[Save] Benchmark results and test subjects saved successfully.")
    except Exception as e:
        print(f"[Error] Failed to save outputs: {e}")


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Modality Machine Learning Baseline")
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the audio dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_audio_ML', help='Directory to save results')
    args = parser.parse_args()

    features_list, labels, groups = extract_opensmile_features(args.data_dir)

    if len(features_list) == 0:
        print("[Terminated] No valid audio data loaded or feature extraction failed.")
    else:
        df_results, latex_snippet, test_subjects = run_audio_ml_single_split_experiments(features_list, labels, groups)
        save_outputs(df_results, latex_snippet, test_subjects, args.output_dir)