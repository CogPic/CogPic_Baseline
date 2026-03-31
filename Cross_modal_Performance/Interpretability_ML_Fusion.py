import os
import datetime
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
# 1. 核心路径配置
# ==========================================
CSV_DIR = r"D:\Code\Project\Dataset\Offline_Features\Handcrafted_CSV"

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = rf"D:\Code\Project\Dataset\Cross_modal_Performance\ML_Benchmark\{current_time}"

TEXT_CSV = os.path.join(CSV_DIR, "Text_Linguistic_Features.csv")
AUDIO_CSV = os.path.join(CSV_DIR, "Audio_Acoustic_Features.csv")
VIDEO_CSV = os.path.join(CSV_DIR, "Video_Facial_Features.csv")

META_COLS = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']


# ==========================================
# 2. 数据加载与预处理引擎
# ==========================================
def load_data(modality="Text"):
    """根据模态名称加载对应的特征表"""
    if modality == "Text":
        df = pd.read_csv(TEXT_CSV)
    elif modality == "Audio":
        df = pd.read_csv(AUDIO_CSV)
    elif modality == "Video":
        df = pd.read_csv(VIDEO_CSV)
    elif modality == "Fused (T+A+V)":
        df_text = pd.read_csv(TEXT_CSV)
        df_audio = pd.read_csv(AUDIO_CSV)
        df_video = pd.read_csv(VIDEO_CSV)
        df = pd.merge(df_text, df_audio, on=META_COLS, how='inner')
        df = pd.merge(df, df_video, on=META_COLS, how='inner')
    else:
        raise ValueError("Unknown modality")

    df = df.fillna(0)
    return df


def prepare_train_test(df):
    """严格按照 Split 划分并分离特征与标签"""
    train_df = df[df['Split'] == 'Train'].reset_index(drop=True)
    test_df = df[df['Split'] == 'Test'].reset_index(drop=True)

    X_train = train_df.drop(columns=META_COLS)
    y_train = train_df['Label_Idx']

    X_test = test_df.drop(columns=META_COLS)
    y_test = test_df['Label_Idx']

    return X_train, y_train, X_test, y_test


# ==========================================
# 3. 核心评估引擎 (特征标准化 + UAR/WAR/AUC)
# ==========================================
def evaluate_ml_baselines(X_train, y_train, X_test, y_test, modality_name):
    # 物理特征必须进行标准化，且只用 train 拟合防止泄露
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 注意：SVC 必须设置 probability=True 才能输出供 AUC 计算的概率值
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

        # 同时获取类别预测和概率预测
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)

        uar = recall_score(y_test, y_pred, average='macro') * 100
        war = recall_score(y_test, y_pred, average='weighted') * 100

        # 兼容处理多分类 AUC
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
# 4. 主流程控制与 LaTeX 表格生成
# ==========================================
def run_comprehensive_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[System] Initiating Comprehensive ML Baseline Benchmark (Aligned Metrics)...")

    modalities_to_test = ["Text", "Audio", "Video", "Fused (T+A+V)"]
    all_results = []

    for mod in modalities_to_test:
        df = load_data(mod)
        X_train, y_train, X_test, y_test = prepare_train_test(df)
        mod_results = evaluate_ml_baselines(X_train, y_train, X_test, y_test, mod)
        all_results.extend(mod_results)

    df_results = pd.DataFrame(all_results)

    csv_out_path = os.path.join(OUTPUT_DIR, "ML_Benchmark_Results.csv")
    df_results.to_csv(csv_out_path, index=False)

    # 生成对齐指标的 LaTeX 表格
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

    txt_out_path = os.path.join(OUTPUT_DIR, "LaTeX_Table_Code.txt")
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    print(f"\n[Process Complete] Benchmark finished successfully.")
    print(f"  -> Detailed CSV saved to: {csv_out_path}")
    print("\n[Generated LaTeX Preview]")
    print(latex_str)


if __name__ == "__main__":
    run_comprehensive_benchmark()