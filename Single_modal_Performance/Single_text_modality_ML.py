import os
import time
import datetime
import pandas as pd
import numpy as np
import warnings
import logging
import gc

# ==========================================
# 0. 环境与日志静音配置
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jieba
import jieba.posseg as pseg

jieba.setLogLevel(logging.ERROR)

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
# 模块 1：真实数据读取与解析
# ==========================================
def load_real_text_data(base_dir):
    label_map = {'HC': 0, 'MCI': 1, 'AD': 2}
    raw_texts = []
    labels = []
    groups = []

    print(f"[数据加载] 开始遍历目录: {base_dir}")

    for label_str, label_idx in label_map.items():
        label_dir = os.path.join(base_dir, label_str)
        if not os.path.exists(label_dir):
            continue

        subject_folders = os.listdir(label_dir)
        for subject_folder in subject_folders:
            subject_dir = os.path.join(label_dir, subject_folder)
            if not os.path.isdir(subject_dir):
                continue

            task_folders = os.listdir(subject_dir)
            for task_folder in task_folders:
                task_dir = os.path.join(subject_dir, task_folder)
                if not os.path.isdir(task_dir):
                    continue

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

    print(f"[数据加载] 完成！共加载 {len(raw_texts)} 条有效文本数据。")
    return raw_texts, np.array(labels), np.array(groups)


# ==========================================
# 模块 2：特征提取 (语言学 + BERT)
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
        if flag.startswith('r'):
            pronoun_count += 1
        elif flag.startswith('n'):
            noun_count += 1

    total_words = len(word_list) if len(word_list) > 0 else 1

    pronoun_ratio = pronoun_count / total_words
    noun_ratio = noun_count / total_words
    ttr = len(set(word_list)) / total_words

    return [avg_sentence_length, pronoun_ratio, noun_ratio, ttr]


def extract_bert_features(text_list, model_name=r'D:\Code\Project\Dataset\Models\bert-base-chinese'):
    print("[特征提取] 正在加载本地 BERT 模型和分词器...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"[错误] 本地 BERT 模型加载失败，请检查路径: {e}")
        return None

    features = []
    print(f"[特征提取] 开始提取 {len(text_list)} 条文本的 BERT 语义特征...")

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
                print(f"  - 已处理 {i + 1}/{len(text_list)} 条文本")

    return np.array(features)


# ==========================================
# 模块 3：模型构建与单次固定划分验证
# ==========================================
def run_baseline_experiments(X, y, groups):
    print("\n[模型评估] 开始执行单次固定划分 (Official Single Fixed Split) 验证...")

    # 核心修改：切出唯一的 1 折作为永久测试集
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    # 提取测试集里的受试者 ID，用于生成官方测试名单
    test_subjects = np.unique(groups[test_idx])
    print(f"  -> 划分完成：训练集包含 {len(train_idx)} 个样本，测试集包含 {len(test_idx)} 个样本 (来自 {len(test_subjects)} 名独立受试者)")

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
            print(f"{name:<25} | {uar:>7.2f}    | {war:>7.2f}    | {auc:>7.2f}    (耗时: {time.time()-start_t:>2.0f}s)")
        except Exception as e:
            print(f"{name:<25} | 运行失败: {str(e)}")

    df_results = pd.DataFrame(results)

    # 渲染纯净版的 LaTeX 代码
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\textbf{UAR (\\%)} & \\textbf{WAR (\\%)} & \\textbf{AUC (\\%)} \\\\\n\\hline\n"
    for _, row in df_results.iterrows():
        auc_text = f"{row['AUC']:.2f}" if not pd.isna(row['AUC']) else "N/A"
        latex_str += f"{row['Model']} & {row['UAR']:.2f} & {row['WAR']:.2f} & {auc_text} \\\\\n"
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Text Modality Machine Learning Baseline (Official 80/20 Fixed Split)}\n\\label{tab:ml_text_baseline}\n\\end{table}"

    return df_results, latex_str, test_subjects


# ==========================================
# 模块 4：输出持久化与官方测试集导出
# ==========================================
def save_outputs(df_results, latex_str, test_subjects, base_output_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_save_path = os.path.join(base_output_path, f"ML_Benchmark_{current_time}")

    try:
        os.makedirs(full_save_path, exist_ok=True)
        print(f"\n[文件保存] 成功创建实验结果目录: {full_save_path}")

        # 保存性能指标
        csv_path = os.path.join(full_save_path, "benchmark_metrics.csv")
        df_results.to_csv(csv_path, index=False)

        # 保存官方测试集受试者名单
        pd.DataFrame({'Subject_ID': test_subjects}).to_csv(os.path.join(full_save_path, "official_test_subjects.csv"), index=False)

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
    DATASET_BASE_DIR = r"D:\Code\Project\Dataset\Full_wly"
    OUTPUT_BASE_DIR = r"D:\Code\Project\Dataset\Single_modal_Performance\Single_text_ML"

    raw_texts, labels, groups = load_real_text_data(DATASET_BASE_DIR)

    if len(raw_texts) == 0:
        print("[终止] 未加载到任何文本数据，请检查目录。")
    else:
        print("\n[阶段一] 开始提取手工语言学特征...")
        linguistic_features = np.array([extract_linguistic_features(text) for text in raw_texts])

        print("\n[阶段二] 开始提取 BERT 深层语义特征...")
        bert_features = extract_bert_features(raw_texts)

        if bert_features is not None:
            print("\n[特征融合] 正在将语言学特征与 BERT 向量进行水平拼接...")
            X_fused = np.hstack((bert_features, linguistic_features))

            print("\n[阶段三] 运行基准模型验证...")
            df_results, latex_snippet, test_subjects = run_baseline_experiments(X_fused, labels, groups)

            print("\n--- 可直接复制的 LaTeX 代码片段 ---")
            print(latex_snippet)
            print("----------------------------------\n")

            save_outputs(df_results, latex_snippet, test_subjects, OUTPUT_BASE_DIR)
        else:
            print("\n[终止] 由于 BERT 特征提取失败，程序已停止。")