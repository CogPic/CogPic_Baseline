import os
import time
import datetime
import subprocess
import pandas as pd
import numpy as np
import warnings
import gc

# ==========================================
# 0. 环境与日志配置
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
# 模块 1：精准目录扫描 (复用 DL 逻辑)
# ==========================================
def scan_video_dataset(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    data_records = []
    groups = []

    print(f"\n[数据加载] 开始精准扫描 120 帧视频目录: {base_dir}")

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
                            'task_id': task_folder,  # 用于生成唯一的文件名
                            'label': label_idx
                        })
                        groups.append(subject_folder)

    print(f"[数据加载] 扫描完成！共定位到 {len(data_records)} 个有效视频任务包。")
    return data_records, np.array(groups)


# ==========================================
# 模块 2：OpenFace 自动化提取与特征降维
# ==========================================
def extract_openface_features(data_records, openface_exe, temp_out_dir):
    """
    调用 OpenFace 提取 AU，并计算全局统计量 (Mean, Std)
    """
    os.makedirs(temp_out_dir, exist_ok=True)
    features_list = []
    valid_indices = []

    print(f"\n[特征工程] 开始通过 OpenFace 提取面部动作单元 (AUs)...")
    print(f"[提示] 此过程可能耗时较长，已开启断点续传缓存机制。")

    for i, record in enumerate(data_records):
        img_dir = record['folder_path']
        task_id = record['task_id']

        # 定义输出的 csv 路径
        out_csv_path = os.path.join(temp_out_dir, f"{task_id}.csv")

        # 1. 检查是否已经提取过 (缓存机制)
        if not os.path.exists(out_csv_path):
            # 构建 OpenFace 命令行指令
            # -fdir: 输入目录; -out_dir: 输出目录; -of: 指定输出文件名; -aus -gaze -pose: 提取指定特征
            cmd = [
                openface_exe,
                '-fdir', img_dir,
                '-out_dir', temp_out_dir,
                '-of', f"{task_id}.csv",
                '-aus', '-gaze', '-pose', '-q'  # -q 为静默模式
            ]

            try:
                # 隐藏 Windows 命令行窗口
                creationflags = 0
                if os.name == 'nt':
                    creationflags = subprocess.CREATE_NO_WINDOW

                # 【核心修复】强行将工作目录(cwd)锁定为 OpenFace 的安装目录
                openface_dir = os.path.dirname(openface_exe)

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=creationflags, check=True, cwd=openface_dir)

            except Exception as e:
                print(f"  [警告] OpenFace 提取失败: {task_id} -> {e}")
                continue

        # 2. 读取 CSV 并进行特征降维 (从 120 帧 -> 1D 向量)
        try:
            if not os.path.exists(out_csv_path):
                continue

            df = pd.read_csv(out_csv_path)
            # OpenFace 的列名带有前导空格，必须清除
            df.columns = df.columns.str.strip()

            # 提取我们关心的特征列
            au_r_cols = [c for c in df.columns if c.startswith('AU') and c.endswith('_r')]  # 动作强度 (0-5)
            au_c_cols = [c for c in df.columns if c.startswith('AU') and c.endswith('_c')]  # 动作发生 (0,1)
            pose_cols = ['pose_Rx', 'pose_Ry', 'pose_Rz']  # 头部旋转
            gaze_cols = ['gaze_angle_x', 'gaze_angle_y']  # 视线角度

            # 计算全局统计量
            feature_vector = []

            # 1. AU 强度的均值和标准差 (反映表情幅度和变化丰富度)
            feature_vector.extend(df[au_r_cols].mean().fillna(0).values)
            feature_vector.extend(df[au_r_cols].std().fillna(0).values)

            # 2. AU 发生的频率 (均值)
            feature_vector.extend(df[au_c_cols].mean().fillna(0).values)

            # 3. 头部姿态的波动 (标准差，反映是否频繁晃动)
            feature_vector.extend(df[pose_cols].std().fillna(0).values)

            # 4. 视线的波动 (标准差，反映眼神是否游离)
            feature_vector.extend(df[gaze_cols].std().fillna(0).values)

            features_list.append(feature_vector)
            valid_indices.append(i)

        except Exception as e:
            print(f"  [警告] 解析 CSV 失败 {out_csv_path}: {e}")
            continue

        if (len(valid_indices)) % 200 == 0:
            print(f"  -> 已成功提取并浓缩 {len(valid_indices)} 个视频的 AU 特征...")

    print(f"[特征工程] 完成！成功生成 {len(features_list)} 个样本的 1D 面部特征向量。")
    return np.array(features_list), valid_indices


# ==========================================
# 模块 3：模型训练与官方基准测试
# ==========================================
def run_video_ml_single_split_experiments(X, y, groups):
    print("\n[模型评估] 执行单次固定划分 (Official Single Fixed Split) 验证...")

    # 严格锁死 random_state=42
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    test_subjects = np.unique(groups[test_idx])
    print(f"  -> 划分完成：训练集 {len(train_idx)} 个，测试集 {len(test_idx)} 个 (来自 {len(test_subjects)} 名受试者)")

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
            print(
                f"{name:<25} | {uar:>7.2f}    | {war:>7.2f}    | {auc:>7.2f}    (耗时: {time.time() - start_t:>2.0f}s)")
        except Exception as e:
            print(f"{name:<25} | 运行失败: {str(e)}")

    df_results = pd.DataFrame(results)

    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Video Modality Action Units ML Baseline (Official 80/20 Fixed Split)}\n\\label{tab:ml_video_au_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# 模块 4：结果持久化
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"Video_ML_Benchmark_{current_time}")

    try:
        os.makedirs(full_save_path, exist_ok=True)
        print(f"\n[文件保存] 成功创建实验结果目录: {full_save_path}")

        df_results.to_csv(os.path.join(full_save_path, "benchmark_metrics.csv"), index=False)
        pd.DataFrame({'Subject_ID': test_subjects}).to_csv(os.path.join(full_save_path, "official_test_subjects.csv"),
                                                           index=False)

        latex_path = os.path.join(full_save_path, "latex_table_code.txt")
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)

        print(f"[文件保存] 官方基准测试结果与受试者名单已安全保存。")
    except Exception as e:
        print(f"[错误] 保存文件时发生异常: {e}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Single_modal_Performance\Single_video_ML"

    # 你指定的 OpenFace 路径 (确保指向 FeatureExtraction.exe)
    OPENFACE_EXE = r"D:\SoftWare\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

    # 缓存特征的临时目录 (强烈建议放在 SSD 上)
    TEMP_OUT_DIR = r"D:\Code\Project\Dataset\OpenFace_Features_Cache"

    data_records, groups_arr = scan_video_dataset(DATASET_BASE_DIR)

    if len(data_records) > 0:
        # 1. 提取或加载 OpenFace 特征
        X_features, valid_indices = extract_openface_features(data_records, OPENFACE_EXE, TEMP_OUT_DIR)

        # 因为可能有极少数视频 OpenFace 无法识别人脸，导致特征提取失败，
        # 我们必须对齐 groups 和 labels
        y_labels = np.array([data_records[i]['label'] for i in valid_indices])
        valid_groups = groups_arr[valid_indices]

        if len(X_features) > 0:
            # 2. 运行机器学习单次划分
            df_results, latex_snippet, test_subjects = run_video_ml_single_split_experiments(X_features, y_labels,
                                                                                             valid_groups)

            print("\n--- 可直接复制的 LaTeX 代码片段 ---")
            print(latex_snippet)
            print("----------------------------------\n")

            save_outputs(df_results, latex_snippet, test_subjects, OUTPUT_BASE_DIR)
        else:
            print("[终止] 所有视频的特征提取均失败，无法进行模型训练。")
    else:
        print("[终止] 未发现任何有效的 120 帧图像数据。")