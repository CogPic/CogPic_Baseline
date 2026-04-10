import os
import datetime
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import recall_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Academic Formatting Configuration
# ==========================================
META_COLS = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']

# Academic black/white/gray styling & Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


# ==========================================
# 2. Modality Routing Intelligence
# ==========================================
def get_modality_category(feature_name):
    """Automatically determine the modality based on feature name keywords."""
    video_prefixes = ('AU', 'gaze', 'pose')
    audio_prefixes = ('f0', 'f1', 'f2', 'f3', 'f4', 'jitter', 'shimmer', 'hnr',
                      'intensity', 'mfcc', 'pause', 'n_pauses', 'avg_pause')
    if feature_name.startswith(video_prefixes):
        return "Video"
    elif feature_name.startswith(audio_prefixes):
        return "Audio"
    else:
        return "Text"


# ==========================================
# 3. Data Loading and Super Fusion
# ==========================================
def load_and_merge_features(text_csv, audio_csv, video_csv):
    print(f"\n[Phase 1] Starting multi-modal feature fusion (Text + Audio + Video)...")

    if not all(os.path.exists(p) for p in [text_csv, audio_csv, video_csv]):
        raise FileNotFoundError("[Fatal Error] One or more input CSV files are missing. Please check your --csv_dir.")

    df_text = pd.read_csv(text_csv)
    df_audio = pd.read_csv(audio_csv)
    df_video = pd.read_csv(video_csv)

    # Inner join across all three modalities using the master meta columns
    df_merged = pd.merge(df_text, df_audio, on=META_COLS, how='inner')
    df_merged = pd.merge(df_merged, df_video, on=META_COLS, how='inner')
    df_merged = df_merged.fillna(0)

    print(f" [Info] Fusion complete. Data shape: {len(df_merged)} rows, "
          f"{df_merged.shape[1] - len(META_COLS)} features.")
    return df_merged


# ==========================================
# 4. Academic SHAP Visualization
# ==========================================
def draw_academic_shap_plot(shap_values_ad, X_test, model_name, output_dir):
    """Generates an academic-grade SHAP beeswarm plot with distinct modality annotations."""

    # Core parameter to control row height and compress vertical spacing
    row_height = 0.18

    # Slightly increase base height to accommodate compressed text
    plt.figure(figsize=(6, 2.8))

    shap.summary_plot(
        shap_values_ad,
        X_test,
        max_display=20,
        show=False,
        cmap='coolwarm',
        plot_size=row_height
    )

    ax = plt.gca()

    # Retain custom label logic (Feature Name + Modality Annotation)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    locs = ax.get_yticks()
    ax.set_yticklabels([])

    # Adjust left margin to prevent text clipping
    plt.subplots_adjust(left=0.58)

    for y, feature_name in zip(locs, labels):
        modality = get_modality_category(feature_name)

        # Feature name text
        ax.text(-0.78, y, feature_name,
                transform=ax.get_yaxis_transform(),
                ha='left', va='center', color='black', fontsize=11.5)

        # Modality tag text
        ax.text(-0.02, y, modality,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', color='black',
                fontsize=12, fontweight='bold')

    plt.title(f"Global Feature Attribution for AD Prediction ({model_name})",
              fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("SHAP value (impact on model output)", fontsize=13)

    # Save outputs
    plot_path_png = os.path.join(output_dir, f"SHAP_Academic_Beeswarm_{model_name.replace(' ', '_')}.png")
    plot_path_pdf = os.path.join(output_dir, f"SHAP_Academic_Beeswarm_{model_name.replace(' ', '_')}.pdf")

    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    print(f" [Output] Saved compressed SHAP plot for {model_name}.")


# ==========================================
# 5. Main Execution: ML Comparison + SHAP
# ==========================================
def run_ml_and_shap(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)

    text_csv = os.path.join(args.csv_dir, "Text_Linguistic_Features.csv")
    audio_csv = os.path.join(args.csv_dir, "Audio_Acoustic_Features.csv")
    video_csv = os.path.join(args.csv_dir, "Video_Facial_Features.csv")

    df_fused = load_and_merge_features(text_csv, audio_csv, video_csv)

    train_df = df_fused[df_fused['Split'] == 'Train'].reset_index(drop=True)
    test_df = df_fused[df_fused['Split'] == 'Test'].reset_index(drop=True)

    X_train = train_df.drop(columns=META_COLS)
    y_train = train_df['Label_Idx']
    X_test = test_df.drop(columns=META_COLS)
    y_test = test_df['Label_Idx']

    # ==================== Safe Data Cleaning ====================
    print(" [Data Cleaning] Converting all features to pure float64...")
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')

    non_numeric = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f" [Warning] Non-numeric columns detected and coerced: {non_numeric}")
    # ============================================================

    # 5 Recommended Classifiers
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            max_depth=4, learning_rate=0.05, n_estimators=150,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            objective='multiclass', num_class=3,
            max_depth=4, learning_rate=0.05, n_estimators=150,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            random_state=42, verbose=False, task_type="CPU"
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
    }

    results = []
    best_model_name = None
    best_uar = -1
    best_clf = None

    print(f"\n[Model Evaluation] Training and evaluating 5 ensemble models...")

    for model_name, clf in models.items():
        print(f"\n--- Evaluating {model_name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        uar = recall_score(y_test, y_pred, average='macro') * 100
        acc = accuracy_score(y_test, y_pred) * 100

        print(f" [Result] {model_name} Test UAR: {uar:.2f}% | Accuracy: {acc:.2f}%")

        results.append({"Model": model_name, "UAR (%)": round(uar, 2), "Accuracy (%)": round(acc, 2)})

        if uar > best_uar:
            best_uar = uar
            best_model_name = model_name
            best_clf = clf

    # Save performance comparison table (for Paper Table 1)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="UAR (%)", ascending=False)
    results_path = os.path.join(output_dir, "Model_Performance_Comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Output] Model performance table saved to: {results_path}")
    print(f" [Best Model] {best_model_name} achieved the highest UAR: {best_uar:.2f}%")

    # ==================== Generate SHAP for Best Model ====================
    print(f"\n[SHAP Analysis] Generating SHAP values for the top-performing model: {best_model_name}...")

    try:
        explainer = shap.TreeExplainer(best_clf)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # Extract AD class for multi-class classification (assuming AD is index 2)
        if isinstance(shap_values, list):
            shap_values_ad = shap_values[2]
        elif len(shap_values.shape) == 3:
            shap_values_ad = shap_values[:, :, 2]
        else:
            shap_values_ad = shap_values

        draw_academic_shap_plot(shap_values_ad, X_test, best_model_name, output_dir)

    except Exception as e:
        print(f" [SHAP Error] {type(e).__name__}: {e}")
        print(" [Tip] If error mentions '[0.5]' or similar, it may be an XGBoost >= 3.0 compatibility issue with SHAP.")
        print("       Consider downgrading XGBoost to < 3.0 or updating SHAP to the latest version.")
        raise

    print(f"\n[Process Complete] All outputs saved to: {output_dir}")
    print(f"   - Performance table: Model_Performance_Comparison.csv")
    print(f"   - Best model SHAP plots (PNG + PDF)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Feature Fusion and SHAP Explanation")
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing the handcrafted modality CSVs')
    parser.add_argument('--output_dir', type=str, default='./outputs/SHAP_Results',
                        help='Base directory to save evaluation metrics and plots')

    args = parser.parse_args()
    run_ml_and_shap(args)