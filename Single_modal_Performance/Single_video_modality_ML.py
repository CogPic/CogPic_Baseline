import os
import time
import datetime
import subprocess
import argparse
import pandas as pd
import numpy as np
import warnings

# ==========================================
# 0. Environment Setup
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# ==========================================
# Module 1: Precise Directory Scanning
# ==========================================
def scan_video_dataset(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    data_records = []
    groups = []

    print(f"\n[Data Loading] Scanning for 120-frame video directories in: {base_dir}")

    for label_str, label_idx in label_map.items():
        label_dir = os.path.join(base_dir, label_str)
        if not os.path.exists(label_dir): continue

        for subject_folder in os.listdir(label_dir):
            subject_dir = os.path.join(label_dir, subject_folder)
            if not os.path.isdir(subject_dir): continue

            for task_folder in os.listdir(subject_dir):
                task_dir = os.path.join(subject_dir, task_folder)
                if not os.path.isdir(task_dir): continue

                target_frames_dir = None
                for sub_item in os.listdir(task_dir):
                    sub_path = os.path.join(task_dir, sub_item)
                    if os.path.isdir(sub_path) and '120' in sub_item:
                        target_frames_dir = sub_path
                        break

                if target_frames_dir is not None:
                    imgs = [f for f in os.listdir(target_frames_dir) if f.endswith('.jpg') or f.endswith('.png')]
                    if len(imgs) > 0:
                        data_records.append({
                            'folder_path': target_frames_dir,
                            'task_id': task_folder,
                            'label': label_idx
                        })
                        groups.append(subject_folder)

    print(f"[Data Loading] Scan complete! Located {len(data_records)} valid video task packages.")
    return data_records, np.array(groups)


# ==========================================
# Module 2: OpenFace Automated Extraction & PCA
# ==========================================
def extract_openface_features(data_records, openface_exe, temp_out_dir):
    os.makedirs(temp_out_dir, exist_ok=True)
    features_list = []
    valid_indices = []

    print(f"\n[Feature Engineering] Extracting Facial Action Units (AUs) via OpenFace...")
    print(f"[Info] Cache mechanism enabled to resume interrupted processes.")

    for i, record in enumerate(data_records):
        img_dir = record['folder_path']
        task_id = record['task_id']
        out_csv_path = os.path.join(temp_out_dir, f"{task_id}.csv")

        # 1. Check Cache
        if not os.path.exists(out_csv_path):
            cmd = [
                openface_exe,
                '-fdir', img_dir,
                '-out_dir', temp_out_dir,
                '-of', f"{task_id}.csv",
                '-aus', '-gaze', '-pose', '-q'
            ]

            try:
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                openface_dir = os.path.dirname(openface_exe)

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=creationflags, check=True, cwd=openface_dir)

            except Exception as e:
                print(f"  [Warning] OpenFace extraction failed for {task_id}: {e}")
                continue

        # 2. Parse CSV and perform dimensionality reduction (120 frames -> 1D vector)
        try:
            if not os.path.exists(out_csv_path): continue

            df = pd.read_csv(out_csv_path)
            df.columns = df.columns.str.strip()

            au_r_cols = [c for c in df.columns if c.startswith('AU') and c.endswith('_r')]
            au_c_cols = [c for c in df.columns if c.startswith('AU') and c.endswith('_c')]
            pose_cols = ['pose_Rx', 'pose_Ry', 'pose_Rz']
            gaze_cols = ['gaze_angle_x', 'gaze_angle_y']

            feature_vector = []
            feature_vector.extend(df[au_r_cols].mean().fillna(0).values)
            feature_vector.extend(df[au_r_cols].std().fillna(0).values)
            feature_vector.extend(df[au_c_cols].mean().fillna(0).values)
            feature_vector.extend(df[pose_cols].std().fillna(0).values)
            feature_vector.extend(df[gaze_cols].std().fillna(0).values)

            features_list.append(feature_vector)
            valid_indices.append(i)

        except Exception as e:
            print(f"  [Warning] Failed to parse CSV {out_csv_path}: {e}")
            continue

        if len(valid_indices) % 200 == 0:
            print(f"  -> Successfully extracted and condensed AU features for {len(valid_indices)} videos...")

    print(f"[Feature Engineering] Complete! Generated 1D facial feature vectors for {len(features_list)} samples.")
    return np.array(features_list), valid_indices


# ==========================================
# Module 3: Model Training & Evaluation
# ==========================================
def run_video_ml_single_split_experiments(X, y, groups):
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
    print(f"\n{'Video ML Model':<25} | {'UAR (%)':<10} | {'WAR (%)':<10} | {'AUC (%)':<10}")
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

    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Video Modality Action Units ML Baseline (Official 80/20 Fixed Split)}\n\\label{tab:ml_video_au_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# Module 4: Output Persistence
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"Video_ML_Benchmark_{current_time}")

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
    parser = argparse.ArgumentParser(description="Video Modality Machine Learning Baseline")
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for the video dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Single_video_ML', help='Directory to save results')
    parser.add_argument('--openface_exe', type=str, required=True, help='Absolute path to OpenFace FeatureExtraction.exe')
    parser.add_argument('--cache_dir', type=str, default='./cache/OpenFace_Features', help='Cache directory for extracted features')
    args = parser.parse_args()

    data_records, groups_arr = scan_video_dataset(args.data_dir)

    if len(data_records) > 0:
        X_features, valid_indices = extract_openface_features(data_records, args.openface_exe, args.cache_dir)

        y_labels = np.array([data_records[i]['label'] for i in valid_indices])
        valid_groups = groups_arr[valid_indices]

        if len(X_features) > 0:
            df_results, latex_snippet, test_subjects = run_video_ml_single_split_experiments(X_features, y_labels, valid_groups)
            save_outputs(df_results, latex_snippet, test_subjects, args.output_dir)
        else:
            print("[Terminated] Feature extraction failed for all videos.")
    else:
        print("[Terminated] No valid 120-frame image data found.")