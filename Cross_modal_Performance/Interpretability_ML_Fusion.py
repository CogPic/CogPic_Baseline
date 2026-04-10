import os
import time
import datetime
import argparse
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. Constants & Meta Configurations
# ==========================================
META_COLS = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']


# ==========================================
# 2. Data Loading & Preprocessing Engine
# ==========================================
def load_data(modality, csv_dir):
    """Loads the corresponding feature table based on the specified modality."""
    text_csv = os.path.join(csv_dir, "Text_Linguistic_Features.csv")
    audio_csv = os.path.join(csv_dir, "Audio_Acoustic_Features.csv")
    video_csv = os.path.join(csv_dir, "Video_Facial_Features.csv")

    # Validate file existence
    expected_files = [text_csv, audio_csv, video_csv]
    if not all(os.path.exists(f) for f in expected_files):
        raise FileNotFoundError(f"[Fatal Error] Missing one or more CSV files in {csv_dir}.")

    if modality == "Text":
        df = pd.read_csv(text_csv)
    elif modality == "Audio":
        df = pd.read_csv(audio_csv)
    elif modality == "Video":
        df = pd.read_csv(video_csv)
    elif modality == "Fused (T+A+V)":
        df_text = pd.read_csv(text_csv)
        df_audio = pd.read_csv(audio_csv)
        df_video = pd.read_csv(video_csv)
        # Inner join across all modalities using meta columns
        df = pd.merge(df_text, df_audio, on=META_COLS, how='inner')
        df = pd.merge(df, df_video, on=META_COLS, how='inner')
    else:
        raise ValueError(f"Unknown modality requested: {modality}")

    df = df.fillna(0)
    return df


def prepare_train_test(df):
    """Splits the dataframe into features and labels strictly based on the 'Split' column."""
    train_df = df[df['Split'] == 'Train'].reset_index(drop=True)
    test_df = df[df['Split'] == 'Test'].reset_index(drop=True)

    X_train = train_df.drop(columns=META_COLS)
    y_train = train_df['Label_Idx']

    X_test = test_df.drop(columns=META_COLS)
    y_test = test_df['Label_Idx']

    return X_train, y_train, X_test, y_test


# ==========================================
# 3. Core Evaluation Engine (Standardization + UAR/WAR/AUC)
# ==========================================
def evaluate_ml_baselines(X_train, y_train, X_test, y_test, modality_name):
    """Trains traditional ML models and calculates performance metrics."""

    # Fit scaler strictly on the training set to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Note: SVC must have probability=True to output probabilities for AUC calculation
    models = {
        "LR": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42,
                                     n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            max_depth=4, learning_rate=0.05, n_estimators=150,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    }

    results = []
    print(f"\n[Evaluating Modality: {modality_name}] Feature Dimension: {X_train.shape[1]}")

    for model_name, clf in models.items():
        clf.fit(X_train_scaled, y_train)

        # Retrieve both class predictions and probability predictions
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)

        uar = recall_score(y_test, y_pred, average='macro') * 100
        war = recall_score(y_test, y_pred, average='weighted') * 100

        # Safely handle multi-class AUC calculation
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro') * 100
        except ValueError:
            auc = np.nan

        auc_text = f"{auc:>5.2f}%" if not pd.isna(auc) else "  N/A"
        print(f"  -> {model_name:<8} | UAR: {uar:>5.2f}% | WAR: {war:>5.2f}% | AUC: {auc_text}")

        results.append({
            "Modality": modality_name,
            "Model": model_name,
            "UAR": uar,
            "WAR": war,
            "AUC": auc
        })

    return results


# ==========================================
# 4. Main Pipeline & LaTeX Generation
# ==========================================
def run_comprehensive_benchmark(csv_dir, output_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, current_time)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[System] Initiating Comprehensive ML Baseline Benchmark (Aligned Metrics)...")

    modalities_to_test = ["Text", "Audio", "Video", "Fused (T+A+V)"]
    all_results = []

    for mod in modalities_to_test:
        df = load_data(mod, csv_dir)
        X_train, y_train, X_test, y_test = prepare_train_test(df)
        mod_results = evaluate_ml_baselines(X_train, y_train, X_test, y_test, mod)
        all_results.extend(mod_results)

    df_results = pd.DataFrame(all_results)

    # Save detailed CSV
    csv_out_path = os.path.join(save_dir, "ML_Benchmark_Results.csv")
    df_results.to_csv(csv_out_path, index=False)

    # Generate aligned LaTeX table
    latex_str = "\\begin{table*}[t]\n\\centering\n\\caption{Performance Comparison of Traditional Machine Learning Models using Handcrafted Features}\n\\label{tab:ml_baselines}\n\\begin{tabular}{llccc}\n\\toprule\n"
    latex_str += "\\textbf{Modality} & \\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\midrule\n"

    current_mod = ""
    for _, row in df_results.iterrows():
        mod_display = row['Modality'] if row['Modality'] != current_mod else ""
        if mod_display != "" and current_mod != "":
            latex_str += "\\midrule\n"
        current_mod = row['Modality']

        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{mod_display} & {row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table*}"

    txt_out_path = os.path.join(save_dir, "LaTeX_Table_Code.txt")
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    print(f"\n[Process Complete] Benchmark finished successfully.")
    print(f"  -> Detailed CSV saved to: {csv_out_path}")
    print("\n[Generated LaTeX Snippet Preview]")
    print(latex_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Multimodal ML Baseline Benchmark")
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing the handcrafted feature CSV files')
    parser.add_argument('--output_dir', type=str, default='./outputs/ML_Benchmark',
                        help='Directory to save benchmark results and LaTeX tables')

    args = parser.parse_args()

    run_comprehensive_benchmark(args.csv_dir, args.output_dir)