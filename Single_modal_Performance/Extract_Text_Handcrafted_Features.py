# ==============================================================================
# 【融合版】AD/MCI/HC 语音文本特征提取系统
# 功能：保留【第一段代码】全部 60-80 维完整特征（流畅性、词汇丰富度、句法复杂度、词性语义、语篇特征、任务相关内容密度、信息单元密度等）
# 输入方式：采用【第二段代码】的官方主清单 CSV 读取方式（Official_Master_Split.csv），支持 Split、Label_Idx 等元数据
# 输出：Text_Linguistic_Features.csv（包含全部特征 + 官方元数据）
# 作者：Grok 为你一键融合优化版
# ==============================================================================

import os
import re
import math
import warnings

# ==============================================================================
# 0. 终极网络防御 + 环境配置（来自第二段，解决 HF 下载问题）
# ==============================================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import jieba
import stanza
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm
# 【新增】第一段需要的 sklearn 模块（用于语篇特征）
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

# ==============================================================================
# 1. 核心配置区（融合两段路径与资源）
# ==============================================================================
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
OUTPUT_DIR = r"D:\Code\Project\Dataset\Offline_Features\Handcrafted_CSV"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "Text_Linguistic_Features.csv")

BASE_DIR_SCI = r"D:\Code\Project\SCI_1"
MARKER_DIR = os.path.join(BASE_DIR_SCI, "Feature extraction", "Discourse_markers", "five_markers")
INFO_UNITS_FILE = Path(BASE_DIR_SCI) / "Feature extraction" / "Semantic Ides Density" / "four_markers" / "The total number of information units.txt"

# 任务核心词汇（两段完全一致）
TASK_VOCAB_MAP = {
    'task_1': {'蛋糕', '洗碗', '盘子', '水龙头', '水池', '柜子', '溢出', '小女孩', '摔倒', '卫生'},
    'task_2': {'放风筝', '钓鱼', '河边', '帆船', '小狗', '饮料', '野外', '海边', '休闲', '旅游', '看书', '放鹞子'},
    'task_3': {'马路', '救护车', '交通事故', '红绿灯', '车祸', '交警', '斑马线', '车', '围观', '现场'}
}

# 中文填充词（两段一致）
CHINESE_FILLED_PAUSES = {
    '嗯', '呃', '啊', '哦', '啦', '诶', '幺', '算算', '好像', '这边', '讲讲', '猜猜', '两点', '二点',
    '是吧', '对吗', '好了', '对吧', '啊啦', '是不是', '对不对', '看不懂', '不知道', '看不出来',
    '看不出来了', '这个', '那个', '就是', '然后'
}

# 【新增】第一段的中文停用词（用于句间余弦相似度）
CHINESE_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
    '说', '去', '你', '会', '着', '没有', '他们', '我们', '他', '她', '它', '被', '比', '吧', '吗', '呢',
    '这', '那', '要', '没', '来', '给', '上面', '两个', '嘛', '应该', '对', '好', '可以', '做', '用'
}

# ==============================================================================
# 2. 辅助资源加载函数（来自第二段，更鲁棒）
# ==============================================================================
def load_discourse_markers(path):
    if not os.path.exists(path):
        return {}
    m = {}
    for f in os.listdir(path):
        if f.endswith('.txt'):
            k = os.path.splitext(f)[0]
            with open(os.path.join(path, f), 'r', encoding='utf-8') as file:
                m[k] = {line.strip() for line in file if line.strip()}
    return m

def load_info_units(path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_task_id_for_vocab(task_string):
    """第二段的智能任务ID映射"""
    if '1' in task_string: return 'task_1'
    if '2' in task_string: return 'task_2'
    if '3' in task_string: return 'task_3'
    return 'task_unknown'

def read_text_robustly(task_dir):
    """第二段的鲁棒读取（utf-8 → gbk 降级）"""
    txt_file_path = None
    if os.path.exists(task_dir):
        for file in os.listdir(task_dir):
            if file.endswith('.txt'):
                txt_file_path = os.path.join(task_dir, file)
                break
    if txt_file_path is None:
        return ""
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(txt_file_path, 'r', encoding='gbk') as f:
                return f.read().strip()
        except:
            return ""

# ==============================================================================
# 3. 核心算法函数库（完整保留第一段全部函数）
# ==============================================================================
def get_dependency_tree_height(sentence):
    """依存句法树高度（第一段原版）"""
    if not sentence or not sentence.words:
        return 0
    nodes = {word.id: [] for word in sentence.words}
    root = None
    for word in sentence.words:
        if word.head == 0:
            root = word.id
        elif word.head in nodes:
            nodes[word.head].append(word.id)
    if root is None:
        return 0
    queue = [(root, 1)]
    max_height = 0
    visited = set()
    while queue:
        curr_id, height = queue.pop(0)
        if curr_id in visited:
            continue
        visited.add(curr_id)
        max_height = max(max_height, height)
        if curr_id in nodes:
            for child_id in nodes[curr_id]:
                queue.append((child_id, height + 1))
    return max_height

def get_constituency_tree_height(tree_node):
    """成分句法树高度（第一段独有）"""
    if not tree_node:
        return 0
    queue = [(tree_node, 1)]
    max_height = 0
    while queue:
        curr_node, curr_height = queue.pop(0)
        max_height = max(max_height, curr_height)
        if curr_node.children:
            for child in curr_node.children:
                queue.append((child, curr_height + 1))
    return max_height

def calculate_vocabulary_richness(tokens):
    """完整词汇丰富度（第一段原版，保留 MATTR）"""
    N = len(tokens)
    V = len(set(tokens))
    res = {}
    if N == 0:
        return {k: 0 for k in ['ttr', 'mattr', 'brunet_w', 'honore_h']}
    res['ttr'] = V / N
    # MATTR（移动平均TTR，窗口50）
    window = 50
    if N < window:
        res['mattr'] = V / N
    else:
        ttr_scores = [len(set(tokens[i:i + window])) / window for i in range(N - window + 1)]
        res['mattr'] = np.mean(ttr_scores)
    res['brunet_w'] = N ** (V ** -0.165) if V > 1 else 0
    counts = Counter(tokens)
    V1 = sum(1 for v in counts.values() if v == 1)
    res['honore_h'] = 100 * (math.log(N) / (1 - (V1 / V))) if V != V1 else 0
    return res

def calculate_fluency(raw_text_list):
    """流畅性（两段一致，保留第一段逻辑）"""
    filled_pauses = 0
    clean_words = []
    for w in raw_text_list:
        if w in CHINESE_FILLED_PAUSES:
            filled_pauses += 1
            continue
        if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', w):
            clean_words.append(w)
    fragments = sum(1 for i in range(len(clean_words) - 1) if clean_words[i] == clean_words[i + 1])
    return filled_pauses, fragments

def calculate_repetitiveness(stanza_doc, stopwords):
    """句间余弦距离（第一段独有语篇特征）"""
    sentences = [' '.join([w.text for w in s.words]) for s in stanza_doc.sentences]
    if len(sentences) < 2:
        return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}
    try:
        vec = CountVectorizer(stop_words=list(stopwords))
        X = vec.fit_transform(sentences)
        if X.shape[1] == 0:
            return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}
        dist_mat = cosine_distances(X)
        upper_tri = dist_mat[np.triu_indices(len(sentences), k=1)]
        if len(upper_tri) == 0:
            return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}
        avg_dist = np.mean(upper_tri)
        high_sim_ratio = np.sum(upper_tri < 0.3) / len(upper_tri)
        return {'avg_cosine_distance': avg_dist, 'highly_similar_pair_ratio': high_sim_ratio}
    except Exception:
        return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}

# ==============================================================================
# 4. 单样本完整特征提取主逻辑（第一段原版 + 第二段任务ID映射）
# ==============================================================================
def extract_features_for_doc(text, stanza_doc, markers, info_units, task_id_str):
    """【融合核心】保留第一段全部特征"""
    feat = {}
    stanza_tokens = [w.text for s in stanza_doc.sentences for w in s.words]
    n_stanza_tokens = len(stanza_tokens)
    if n_stanza_tokens == 0:
        return None

    jieba_tokens = jieba.lcut(text)
    n_jieba_tokens = max(len(jieba_tokens), 1)

    # --- A. 基础计数 ---
    feat['total_chars'] = len(text)
    feat['total_words'] = n_stanza_tokens

    # --- B. 任务相关内容密度（核心）---
    mapped_task_id = get_task_id_for_vocab(task_id_str)
    target_vocab = TASK_VOCAB_MAP.get(mapped_task_id, set())
    relevant_count = sum(1 for w in jieba_tokens if w in target_vocab)
    feat['task_relevant_content_ratio'] = relevant_count / n_jieba_tokens if target_vocab else 0

    # --- C. 流畅性 ---
    fps, frags = calculate_fluency(jieba_tokens)
    feat['filled_pause_count'] = fps
    feat['filled_pause_ratio'] = fps / n_jieba_tokens
    feat['word_fragment_count'] = frags
    feat['word_fragment_ratio'] = frags / n_jieba_tokens

    # --- D. 词汇丰富度（含 MATTR）---
    richness = calculate_vocabulary_richness(stanza_tokens)
    feat.update(richness)

    # --- E. 词性与语义（完整10类 + 派生比率）---
    pos_tags = [w.upos for s in stanza_doc.sentences for w in s.words]
    n_pos = max(len(pos_tags), 1)
    pos_counts = Counter(pos_tags)
    target_pos_types = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'AUX', 'NUM', 'PART', 'CCONJ']
    for p in target_pos_types:
        c = pos_counts.get(p, 0)
        feat[f'pos_count_{p}'] = c
        feat[f'pos_ratio_{p}'] = c / n_pos
    noun_c = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    verb_c = pos_counts.get('VERB', 0)
    pron_c = pos_counts.get('PRON', 0)
    content_c = noun_c + verb_c + pos_counts.get('ADJ', 0) + pos_counts.get('ADV', 0)
    func_c = n_pos - content_c
    feat['noun_to_verb_ratio'] = noun_c / verb_c if verb_c > 0 else 0
    feat['pronoun_to_noun_ratio'] = pron_c / noun_c if noun_c > 0 else 0
    feat['content_to_function_ratio'] = content_c / func_c if func_c > 0 else 0
    feat['syntactic_idea_density'] = (verb_c + pos_counts.get('ADJ', 0) + pos_counts.get('ADV', 0)) / n_pos

    # --- F. 句法复杂度（依存 + 成分树全部保留）---
    num_sents = len(stanza_doc.sentences)
    if num_sents > 0:
        # 依存句法
        dep_heights = [get_dependency_tree_height(s) for s in stanza_doc.sentences]
        feat['dependency_mean_height'] = np.mean(dep_heights)
        feat['dependency_max_height'] = np.max(dep_heights)
        all_deprels = [w.deprel for s in stanza_doc.sentences for w in s.words]
        dep_counts = Counter(all_deprels)
        for rel in ['nsubj', 'obj', 'advmod', 'amod']:
            c = dep_counts.get(rel, 0)
            feat[f'deprel_count_{rel}'] = c
            feat[f'deprel_ratio_{rel}'] = c / n_stanza_tokens
        dists = [abs(w.id - w.head) for s in stanza_doc.sentences for w in s.words if w.head != 0]
        feat['mean_dependency_distance'] = np.mean(dists) if dists else 0

        # 成分句法（第一段独有）
        const_heights = [get_constituency_tree_height(s.constituency) for s in stanza_doc.sentences if s.constituency]
        if const_heights:
            feat['constituency_mean_height'] = np.mean(const_heights)
            feat['constituency_max_height'] = np.max(const_heights)
            feat['constituency_std_height'] = np.std(const_heights)
            feat['constituency_median_height'] = np.median(const_heights)
        else:
            feat['constituency_mean_height'] = feat['constituency_max_height'] = \
                feat['constituency_std_height'] = feat['constituency_median_height'] = 0
    else:
        feat['dependency_mean_height'] = feat['dependency_max_height'] = \
            feat['mean_dependency_distance'] = 0
        feat['constituency_mean_height'] = feat['constituency_max_height'] = \
            feat['constituency_std_height'] = feat['constituency_median_height'] = 0

    # --- G. 语篇与语义（话语标记 + 句间跳跃 + 信息单元）---
    # 话语标记
    if markers:
        for cat, words in markers.items():
            c = sum(1 for w in stanza_tokens if w in words)
            feat[f'marker_count_{cat}'] = c
            feat[f'marker_ratio_{cat}'] = c / n_stanza_tokens
    # 内容重复性 / 语义跳跃
    rep_feats = calculate_repetitiveness(stanza_doc, CHINESE_STOPWORDS)
    feat.update(rep_feats)
    # 信息单元密度
    if info_units:
        unit_hits = sum(text.count(unit) for unit in info_units)
        feat['semantic_info_unit_count'] = unit_hits
        feat['semantic_idea_density'] = unit_hits / n_stanza_tokens if n_stanza_tokens > 0 else 0

    return feat

# ==============================================================================
# 5. 主流程（第二段的 CSV + tqdm 风格 + 第一段完整特征）
# ==============================================================================
def run_text_handcrafted_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n>>> [环境准备] 正在加载 Stanza 完整句法模型（tokenize,pos,lemma,depparse,constituency）...")
    print("    (首次运行会从国内镜像下载，请耐心等待...)")
    try:
        nlp = stanza.Pipeline(
            lang='zh',
            processors='tokenize,pos,lemma,depparse,constituency',
            use_gpu=torch.cuda.is_available(),
            logging_level='WARN'
        )
    except Exception as e:
        print(f"\n[致命错误] Stanza 加载失败: {e}")
        return

    print(">>> [环境准备] 正在注册任务专属词汇到 Jieba...")
    count_added = 0
    for task_words in TASK_VOCAB_MAP.values():
        for word in task_words:
            jieba.add_word(word)
            count_added += 1
    print(f"    成功注册 {count_added} 个专业词汇。")

    # 加载外部资源（话语标记 + 信息单元）
    markers = load_discourse_markers(MARKER_DIR)
    info_units = load_info_units(INFO_UNITS_FILE)
    if markers:
        print(f">>> 已加载 {len(markers)} 类话语标记。")
    if info_units:
        print(f">>> 已加载 {len(info_units)} 条信息单元。")

    try:
        master_df = pd.read_csv(CSV_PATH)
        print(f"\n>>> [启动] 成功加载官方主清单，共 {len(master_df)} 个样本。")
    except Exception as e:
        print(f"[致命错误] 无法读取主清单: {e}")
        return

    extracted_records = []
    pbar = tqdm(total=len(master_df), desc=" 完整语言特征提取中（60-80维）", unit="text", ncols=100)

    for index, row in master_df.iterrows():
        text = read_text_robustly(row['Task_Dir'])
        # 官方元数据（来自第二段）
        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': row['Task_ID'],
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if text and len(text) > 0:
            try:
                stanza_doc = nlp(text)
                ling_features = extract_features_for_doc(
                    text=text,
                    stanza_doc=stanza_doc,
                    markers=markers,
                    info_units=info_units,
                    task_id_str=row['Task_ID']
                )
                if ling_features:
                    record.update(ling_features)
            except Exception as e:
                print(f"  [警告] 样本 {row['Subject_ID']} 处理异常: {e}")
                pass

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    # 生成 DataFrame 并美化列顺序
    df_features = pd.DataFrame(extracted_records).fillna(0)
    meta_cols = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']
    feat_cols = sorted([c for c in df_features.columns if c not in meta_cols])
    df_features = df_features[meta_cols + feat_cols]

    df_features.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n[大功告成] 融合版特征提取完毕！")
    print(f" → 共生成 {len(df_features)} 行记录")
    print(f" → 特征总维度: {len(feat_cols)}（完整版 60-80 维）")
    print(f" → 文件保存至: {OUTPUT_CSV_PATH}")
    print(f" → 前 5 行预览：")
    print(df_features.head())

if __name__ == "__main__":
    run_text_handcrafted_extraction()