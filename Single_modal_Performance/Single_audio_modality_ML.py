import os
import time
import datetime
import pandas as pd
import numpy as np
import warnings
import gc

# ==========================================
# 0. 环境与日志配置
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import opensmile
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from collections import Counter


# ==========================================
# 模块 1：音频特征提取 (OpenSMILE eGeMAPS)
# ==========================================
def extract_opensmile_features(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    features_list = []
    labels = []
    groups = []

    print(f"\n[数据加载] 开始遍历音频目录: {base_dir}")
    print("[提示] 正在初始化 OpenSMILE 引擎 (特征集: eGeMAPSv02, 级别: Functionals)...")

    # 初始化 OpenSMILE 引擎
    # eGeMAPSv02 包含 88 维专家声学特征，Functionals 表示对整段变长音频进行统计计算(均值、方差等)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    success_count = 0
    for label_str, label_idx in label_map.items():
        label_dir = os.path.join(base_dir, label_str)
        if not os.path.exists(label_dir):
            continue

        for subject_folder in os.listdir(label_dir):
            subject_dir = os.path.join(label_dir, subject_folder)
            if not os.path.isdir(subject_dir):
                continue

            for task_folder in os.listdir(subject_dir):
                task_dir = os.path.join(subject_dir, task_folder)
                if not os.path.isdir(task_dir):
                    continue

                for file in os.listdir(task_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(task_dir, file)
                        try:
                            # 提取 88 维特征，返回一个 DataFrame，取第一行的 values
                            feature_vector = smile.process_file(file_path).values[0]

                            if feature_vector is not None and len(feature_vector) == 88:
                                features_list.append(feature_vector)
                                labels.append(label_idx)
                                groups.append(subject_folder)
                                success_count += 1

                                if success_count % 300 == 0:
                                    print(f"  -> 已成功提取 {success_count} 个音频的 eGeMAPS 特征...")
                        except Exception as e:
                            print(f"  [警告] 提取音频特征失败 {file_path}: {e}")

    print(f"[数据加载] 完成！共成功提取 {len(features_list)} 个音频的 88 维声学特征。")
    return np.array(features_list), np.array(labels), np.array(groups)


# ==========================================
# 模块 2：模型构建与单次固定划分验证
# ==========================================
# ==========================================
# 模块 2：模型构建与单次固定 8:2 划分验证（按受试者分层）
# ==========================================
def run_audio_ml_single_split_experiments(X, y, groups):
    print("\n[模型评估] 开始执行单次固定 8:2 划分 (Official 80/20 Stratified Subject Split)...")

    # ====================== 核心修改部分：按受试者级别 8:2 分层划分 ======================
    unique_subjects = np.unique(groups)
    subject_labels = []

    # 为每个受试者确定其类别（取该受试者所有样本中最多的标签，通常一个受试者只属于一类）
    for subj in unique_subjects:
        subj_mask = (groups == subj)
        subj_y = y[subj_mask]
        most_common_label = Counter(subj_y).most_common(1)[0][0]
        subject_labels.append(most_common_label)

    subject_labels = np.array(subject_labels)

    # 按受试者级别进行分层抽样（保证三类比例均衡）
    from sklearn.model_selection import train_test_split
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=0.2,  # 严格 20% 的受试者作为测试集
        stratify=subject_labels,
        random_state=42  # 固定随机种子，保证可复现
    )

    # 根据受试者ID得到对应的样本索引
    train_idx = np.isin(groups, train_subjects)
    test_idx = np.isin(groups, test_subjects)

    # ====================== 打印划分信息 ======================
    print(f"  -> 划分完成：")
    print(f"     总受试者数      : {len(unique_subjects)} 人")
    print(f"     训练集受试者    : {len(train_subjects)} 人")
    print(f"     测试集受试者    : {len(test_subjects)} 人")
    print(f"     训练集样本数    : {len(train_idx)} 个")
    print(f"     测试集样本数    : {len(test_idx)} 个")

    # 测试集各类别受试者分布（这是你最关心的均衡性）
    print("\n  === 测试集受试者类别分布 ===")
    test_subj_labels = subject_labels[np.isin(unique_subjects, test_subjects)]
    for cls_name, cls_idx in [('HC', 0), ('MCI', 1), ('AD', 2)]:
        count = np.sum(test_subj_labels == cls_idx)
        ratio = count / len(test_subjects) * 100
        print(f"     {cls_name:>3} : {count:3d} 人 ({ratio:5.1f}%)")

    # 提取实际训练集和测试集
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ====================== 特征标准化 ======================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ====================== 定义模型 ======================
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
            print(
                f"{name:<25} | {uar:>7.2f}    | {war:>7.2f}    | {auc:>7.2f}    (耗时: {time.time() - start_t:>2.0f}s)")
        except Exception as e:
            print(f"{name:<25} | 运行失败: {str(e)}")

    df_results = pd.DataFrame(results)

    # ====================== 生成 LaTeX 表格 ======================
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Audio Modality eGeMAPS ML Baseline (Official 80/20 Stratified Subject Split)}\n\\label{tab:ml_audio_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# 模块 3：输出持久化与官方测试集导出
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"Audio_ML_Benchmark_{current_time}")

    try:
        os.makedirs(full_save_path, exist_ok=True)
        print(f"\n[文件保存] 成功创建实验结果目录: {full_save_path}")

        # 保存性能指标
        csv_path = os.path.join(full_save_path, "benchmark_metrics.csv")
        df_results.to_csv(csv_path, index=False)

        # 保存官方测试集受试者名单
        pd.DataFrame({'Subject_ID': test_subjects}).to_csv(os.path.join(full_save_path, "official_test_subjects.csv"),
                                                           index=False)

        # 保存 LaTeX 表格
        latex_path = os.path.join(full_save_path, "latex_table_code.txt")
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)

        print(f"[文件保存] 官方基准测试结果与受试者名单已安全保存。")

    except Exception as e:
        print(f"[错误] 创建文件夹或保存文件时发生异常: {e}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 你的音频数据根目录
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    # 音频机器学习结果保存目录
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Single_modal_Performance\Single_audio_ML"

    features_list, labels, groups = extract_opensmile_features(DATASET_BASE_DIR)

    if len(features_list) == 0:
        print("[终止] 未加载到任何有效音频数据或特征提取失败，请检查目录。")
    else:
        df_results, latex_snippet, test_subjects = run_audio_ml_single_split_experiments(features_list, labels, groups)

        print("\n--- 可直接复制的 LaTeX 代码片段 ---")
        print(latex_snippet)
        print("----------------------------------\n")

        save_outputs(df_results, latex_snippet, test_subjects, OUTPUT_BASE_DIR)