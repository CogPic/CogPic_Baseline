import os
import re
import math
import warnings
import argparse

# ==============================================================================
# 0. Environment Setup
# ==============================================================================
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Uncomment for mainland China
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

# ==============================================================================
# 1. Core NLP Configuration (Chinese Specific Dictionaries)
# ==============================================================================
# Task-specific vocabulary for calculating content density
TASK_VOCAB_MAP = {
    'task_1': {'蛋糕', '洗碗', '盘子', '水龙头', '水池', '柜子', '溢出', '小女孩', '摔倒', '卫生'},
    'task_2': {'放风筝', '钓鱼', '河边', '帆船', '小狗', '饮料', '野外', '海边', '休闲', '旅游', '看书', '放鹞子'},
    'task_3': {'马路', '救护车', '交通事故', '红绿灯', '车祸', '交警', '斑马线', '车', '围观', '现场'}
}

# Chinese filled pauses for fluency calculation
CHINESE_FILLED_PAUSES = {
    '嗯', '呃', '啊', '哦', '啦', '诶', '幺', '算算', '好像', '这边', '讲讲', '猜猜', '两点', '二点',
    '是吧', '对吗', '好了', '对吧', '啊啦', '是不是', '对不对', '看不懂', '不知道', '看不出来',
    '看不出来了', '这个', '那个', '就是', '然后'
}

# Chinese stopwords for inter-sentence cosine similarity
CHINESE_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
    '说', '去', '你', '会', '着', '没有', '他们', '我们', '他', '她', '它', '被', '比', '吧', '吗', '呢',
    '这', '那', '要', '没', '来', '给', '上面', '两个', '嘛', '应该', '对', '好', '可以', '做', '用'
}


# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def load_discourse_markers(path):
    if not path or not os.path.exists(path):
        return {}
    m = {}
    for f in os.listdir(path):
        if f.endswith('.txt'):
            k = os.path.splitext(f)[0]
            with open(os.path.join(path, f), 'r', encoding='utf-8') as file:
                m[k] = {line.strip() for line in file if line.strip()}
    return m


def load_info_units(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_task_id_for_vocab(task_string):
    if '1' in task_string: return 'task_1'
    if '2' in task_string: return 'task_2'
    if '3' in task_string: return 'task_3'
    return 'task_unknown'


def build_path_mapping(base_dir):
    """Dynamically map Task_ID to .txt paths."""
    mapping = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                task_id = os.path.basename(root)
                mapping[task_id] = os.path.join(root, file)
    return mapping


def read_text_robustly(txt_file_path):
    """Robust text reading with fallback encoding."""
    if not txt_file_path or not os.path.exists(txt_file_path):
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
# 3. Core Algorithm Functions
# ==============================================================================
def get_dependency_tree_height(sentence):
    if not sentence or not sentence.words: return 0
    nodes = {word.id: [] for word in sentence.words}
    root = None
    for word in sentence.words:
        if word.head == 0:
            root = word.id
        elif word.head in nodes:
            nodes[word.head].append(word.id)
    if root is None: return 0

    queue = [(root, 1)]
    max_height = 0
    visited = set()
    while queue:
        curr_id, height = queue.pop(0)
        if curr_id in visited: continue
        visited.add(curr_id)
        max_height = max(max_height, height)
        if curr_id in nodes:
            for child_id in nodes[curr_id]:
                queue.append((child_id, height + 1))
    return max_height


def get_constituency_tree_height(tree_node):
    if not tree_node: return 0
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
    N, V = len(tokens), len(set(tokens))
    res = {}
    if N == 0: return {k: 0 for k in ['ttr', 'mattr', 'brunet_w', 'honore_h']}

    res['ttr'] = V / N
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
    sentences = [' '.join([w.text for w in s.words]) for s in stanza_doc.sentences]
    if len(sentences) < 2: return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}

    try:
        vec = CountVectorizer(stop_words=list(stopwords))
        X = vec.fit_transform(sentences)
        if X.shape[1] == 0: return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}

        dist_mat = cosine_distances(X)
        upper_tri = dist_mat[np.triu_indices(len(sentences), k=1)]
        if len(upper_tri) == 0: return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}

        return {
            'avg_cosine_distance': np.mean(upper_tri),
            'highly_similar_pair_ratio': np.sum(upper_tri < 0.3) / len(upper_tri)
        }
    except Exception:
        return {'avg_cosine_distance': 0, 'highly_similar_pair_ratio': 0}


# ==============================================================================
# 4. Main Extraction Logic
# ==============================================================================
def extract_features_for_doc(text, stanza_doc, markers, info_units, task_id_str):
    feat = {}
    stanza_tokens = [w.text for s in stanza_doc.sentences for w in s.words]
    n_stanza_tokens = len(stanza_tokens)
    if n_stanza_tokens == 0: return None

    jieba_tokens = jieba.lcut(text)
    n_jieba_tokens = max(len(jieba_tokens), 1)

    # A. Basic Counts
    feat['total_chars'] = len(text)
    feat['total_words'] = n_stanza_tokens

    # B. Task-Relevant Content Density
    mapped_task_id = get_task_id_for_vocab(task_id_str)
    target_vocab = TASK_VOCAB_MAP.get(mapped_task_id, set())
    relevant_count = sum(1 for w in jieba_tokens if w in target_vocab)
    feat['task_relevant_content_ratio'] = relevant_count / n_jieba_tokens if target_vocab else 0

    # C. Fluency
    fps, frags = calculate_fluency(jieba_tokens)
    feat['filled_pause_count'] = fps
    feat['filled_pause_ratio'] = fps / n_jieba_tokens
    feat['word_fragment_count'] = frags
    feat['word_fragment_ratio'] = frags / n_jieba_tokens

    # D. Vocabulary Richness
    feat.update(calculate_vocabulary_richness(stanza_tokens))

    # E. POS and Semantics
    pos_tags = [w.upos for s in stanza_doc.sentences for w in s.words]
    n_pos = max(len(pos_tags), 1)
    pos_counts = Counter(pos_tags)

    for p in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'AUX', 'NUM', 'PART', 'CCONJ']:
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

    # F. Syntactic Complexity
    if len(stanza_doc.sentences) > 0:
        dep_heights = [get_dependency_tree_height(s) for s in stanza_doc.sentences]
        feat['dependency_mean_height'] = np.mean(dep_heights)
        feat['dependency_max_height'] = np.max(dep_heights)

        dep_counts = Counter([w.deprel for s in stanza_doc.sentences for w in s.words])
        for rel in ['nsubj', 'obj', 'advmod', 'amod']:
            c = dep_counts.get(rel, 0)
            feat[f'deprel_count_{rel}'] = c
            feat[f'deprel_ratio_{rel}'] = c / n_stanza_tokens

        dists = [abs(w.id - w.head) for s in stanza_doc.sentences for w in s.words if w.head != 0]
        feat['mean_dependency_distance'] = np.mean(dists) if dists else 0

        const_heights = [get_constituency_tree_height(s.constituency) for s in stanza_doc.sentences if s.constituency]
        if const_heights:
            feat['constituency_mean_height'] = np.mean(const_heights)
            feat['constituency_max_height'] = np.max(const_heights)
            feat['constituency_std_height'] = np.std(const_heights)
            feat['constituency_median_height'] = np.median(const_heights)
        else:
            feat['constituency_mean_height'] = feat['constituency_max_height'] = feat['constituency_std_height'] = feat[
                'constituency_median_height'] = 0
    else:
        feat['dependency_mean_height'] = feat['dependency_max_height'] = feat['mean_dependency_distance'] = 0
        feat['constituency_mean_height'] = feat['constituency_max_height'] = feat['constituency_std_height'] = feat[
            'constituency_median_height'] = 0

    # G. Discourse and Semantics
    if markers:
        for cat, words in markers.items():
            c = sum(1 for w in stanza_tokens if w in words)
            feat[f'marker_count_{cat}'] = c
            feat[f'marker_ratio_{cat}'] = c / n_stanza_tokens

    feat.update(calculate_repetitiveness(stanza_doc, CHINESE_STOPWORDS))

    if info_units:
        unit_hits = sum(text.count(unit) for unit in info_units)
        feat['semantic_info_unit_count'] = unit_hits
        feat['semantic_idea_density'] = unit_hits / n_stanza_tokens if n_stanza_tokens > 0 else 0

    return feat


# ==============================================================================
# 5. Pipeline Execution
# ==============================================================================
def run_text_handcrafted_extraction(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "Text_Linguistic_Features.csv")

    print("\n>>> [Setup] Initializing Stanza NLP Pipeline (tokenize,pos,lemma,depparse,constituency)...")
    try:
        nlp = stanza.Pipeline(
            lang='zh',
            processors='tokenize,pos,lemma,depparse,constituency',
            use_gpu=torch.cuda.is_available(),
            logging_level='WARN'
        )
    except Exception as e:
        print(f"\n[Fatal Error] Stanza initialization failed: {e}")
        return

    print(">>> [Setup] Registering Task-Specific Vocabularies to Jieba...")
    for task_words in TASK_VOCAB_MAP.values():
        for word in task_words:
            jieba.add_word(word)

    markers = load_discourse_markers(args.marker_dir)
    info_units = load_info_units(args.info_units_file)

    try:
        master_df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[Fatal Error] Failed to read master CSV: {e}")
        return

    path_mapping = build_path_mapping(args.data_dir)
    extracted_records = []

    pbar = tqdm(total=len(master_df), desc="Extracting Linguistic Features", unit="text", ncols=100)

    for _, row in master_df.iterrows():
        task_id = str(row['Task_ID']).strip()
        txt_path = path_mapping.get(task_id)
        text = read_text_robustly(txt_path)

        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': task_id,
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if text and len(text) > 0:
            try:
                stanza_doc = nlp(text)
                ling_features = extract_features_for_doc(text, stanza_doc, markers, info_units, task_id)
                if ling_features:
                    record.update(ling_features)
            except Exception as e:
                print(f"  [Warning] Processing error for {row['Subject_ID']}: {e}")

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    df_features = pd.DataFrame(extracted_records).fillna(0)
    meta_cols = ['Subject_ID', 'Task_ID', 'Label_Str', 'Label_Idx', 'Split']
    feat_cols = sorted([c for c in df_features.columns if c not in meta_cols])
    df_features = df_features[meta_cols + feat_cols]
    df_features.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[Success] Linguistic Feature Extraction Complete!")
    print(f" -> Total records: {len(df_features)}")
    print(f" -> Total feature dimensions: {len(feat_cols)}")
    print(f" -> Output saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Linguistic Feature Extraction")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for text dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Handcrafted_CSV', help='Output directory')
    parser.add_argument('--marker_dir', type=str, default=None, help='Directory containing discourse markers txt files')
    parser.add_argument('--info_units_file', type=str, default=None, help='Path to info units txt file')

    args = parser.parse_args()
    run_text_handcrafted_extraction(args)