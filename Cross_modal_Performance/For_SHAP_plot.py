import os
import datetime
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
# 1. 核心路径与学术排版配置
# ==========================================
CSV_DIR = r"D:\Code\Project\Dataset\Offline_Features\Handcrafted_CSV"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = rf"D:\Code\Project\Dataset\Cross_modal_Performance\SHAP\{current_time}"

TEXT_CSV = os.path.join(CSV_DIR, "Text_Linguistic_Features.csv")
AUDIO_CSV = os.path.join(CSV_DIR, "Audio_Acoustic_Features.csv")
VIDEO_CSV = os.path.join(CSV_DIR, "Video_Facial_Features.csv")

META_COLS = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']

# 学术黑白灰排版风格 & Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


# ==========================================
# 2. 模态智能识别路由
# ==========================================
def get_modality_category(feature_name):
    """根据特征名称的关键字，自动判断其所属模态"""
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
# 3. 数据加载与超级融合
# ==========================================
def load_and_merge_features():
    print(f"\n[Phase 3] Starting multi-modal feature fusion (Text + Audio + Video)...")
    df_text = pd.read_csv(TEXT_CSV)
    df_audio = pd.read_csv(AUDIO_CSV)
    df_video = pd.read_csv(VIDEO_CSV)

    df_merged = pd.merge(df_text, df_audio, on=META_COLS, how='inner')
    df_merged = pd.merge(df_merged, df_video, on=META_COLS, how='inner')
    df_merged = df_merged.fillna(0)

    print(f" [Info] Fusion complete. Data shape: {len(df_merged)} rows, "
          f"{df_merged.shape[1] - len(META_COLS)} features.")
    return df_merged

# ==========================================
# 4. 纯学术风格 SHAP 渲染引擎（紧凑排版优化版）
# ==========================================
def draw_academic_shap_plot(shap_values_ad, X_test, model_name):
    """绘制带有独立模态列的学术级英文 SHAP 蜂窝图 (紧凑版)"""
    plt.figure(figsize=(11, 6))

    shap.summary_plot(
        shap_values_ad, X_test, max_display=15, show=False, cmap='coolwarm'
    )

    ax = plt.gca()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    locs = ax.get_yticks()

    ax.set_yticklabels([])

    plt.subplots_adjust(left=0.56)

    for y, feature_name in zip(locs, labels):
        modality = get_modality_category(feature_name)

        # 【修改 3】调整文字的 X 坐标与字号，适应新的紧凑画布
        # 特征名（贴近左侧边缘）
        ax.text(-0.70, y, feature_name,
                transform=ax.get_yaxis_transform(),
                ha='left', va='center',
                color='black', fontsize=11)  # 字号微调为 11 显得更精致

        # 模态标签（靠近纵轴，加大与特征名的区分度）
        ax.text(-0.03, y, modality,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center',
                color='black', fontsize=11, fontweight='bold')

    # 标题和坐标轴标签也适当调小一点，配合整体紧凑感
    plt.title(f"Global Feature Attribution for AD Prediction ({model_name})",
              fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("SHAP value (impact on model output)", fontsize=12)

    plot_path_png = os.path.join(
        OUTPUT_DIR, f"SHAP_Academic_Beeswarm_{model_name.replace(' ', '_')}.png"
    )
    plot_path_pdf = os.path.join(
        OUTPUT_DIR, f"SHAP_Academic_Beeswarm_{model_name.replace(' ', '_')}.pdf"
    )

    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f" [Output] Saved SHAP plot for {model_name} (PNG & PDF).")


# ==========================================
# 5. 主流程：5个模型对比 + 最优模型 SHAP（安全版，无猴子补丁）
# ==========================================
def run_ml_and_shap():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_fused = load_and_merge_features()

    train_df = df_fused[df_fused['Split'] == 'Train'].reset_index(drop=True)
    test_df = df_fused[df_fused['Split'] == 'Test'].reset_index(drop=True)

    X_train = train_df.drop(columns=META_COLS)
    y_train = train_df['Label_Idx']
    X_test = test_df.drop(columns=META_COLS)
    y_test = test_df['Label_Idx']

    # ==================== 安全数据清洗（关键修复） ====================
    print(" [Data Cleaning] Converting all features to pure float64...")
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')

    # 检查是否仍有非数值列（调试用）
    non_numeric = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f" [Warning] Still has non-numeric columns: {non_numeric}")
    # ============================================================

    # 5个推荐模型
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

    print(f"\n[Model Evaluation] Training and evaluating 5 models...")

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

    # 保存性能对比表格（用于论文 Table 1）
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="UAR (%)", ascending=False)
    results_path = os.path.join(OUTPUT_DIR, "Model_Performance_Comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Output] Model performance table saved to: {results_path}")
    print(f" [Best Model] {best_model_name} with UAR: {best_uar:.2f}%")

    # ==================== 只在最优模型上运行 SHAP（安全版） ====================
    print(f"\n[SHAP Analysis] Generating SHAP values for the best model: {best_model_name}...")

    try:
        explainer = shap.TreeExplainer(best_clf)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # 多分类时取 AD 类（请根据你的 Label_Idx 确认索引，通常 AD 为 2）
        if isinstance(shap_values, list):
            shap_values_ad = shap_values[2]
        elif len(shap_values.shape) == 3:
            shap_values_ad = shap_values[:, :, 2]
        else:
            shap_values_ad = shap_values

        draw_academic_shap_plot(shap_values_ad, X_test, best_model_name)

    except Exception as e:
        print(f" [SHAP Error] {type(e).__name__}: {e}")
        print(
            " [Tip] If error mentions '[0.5]' or similar, it may be XGBoost >= 3.0 compatibility issue with current SHAP.")
        print("       Consider downgrading XGBoost to < 3.0 or updating SHAP to latest version.")
        raise

    print(f"\n[Process Complete] All outputs saved to: {OUTPUT_DIR}")
    print(f"   - Performance table: Model_Performance_Comparison.csv")
    print(f"   - Best model SHAP plots (PNG + PDF)")


if __name__ == "__main__":
    run_ml_and_shap()