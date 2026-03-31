import os
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.stats import skew, kurtosis
import warnings
from tqdm import tqdm

# 忽略常规的除以0或音频较短的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心路径配置
# ==========================================
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
OUTPUT_DIR = r"D:\Code\Project\Dataset\Offline_Features\Handcrafted_CSV"
# 最终生成的结构化声学特征表
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "Audio_Acoustic_Features.csv")

N_MFCC = 42
MIN_SAMPLES_FOR_FFT = 2048


# ==========================================
# 2. 辅助函数：健壮的音频寻址与空特征生成
# ==========================================
def find_wav_robustly(task_dir):
    """在任务文件夹中寻找 .wav 文件"""
    if not os.path.exists(task_dir): return None
    for file in os.listdir(task_dir):
        if file.endswith('.wav'):
            return os.path.join(task_dir, file)
    return None


def get_empty_acoustic_features():
    """当音频损坏或太短时，返回全 0 的特征字典，保证列对齐"""
    feat = {
        'f0_mean': 0, 'f0_std': 0, 'f0_max': 0, 'f0_min': 0,
        'jitter_local': 0, 'shimmer_local': 0, 'hnr_mean': 0, 'intensity_mean': 0,
        'pause_duration': 0, 'n_pauses': 0, 'pause_rate': 0, 'avg_pause_duration': 0,
        'mfcc_mean_vec_skew': 0, 'mfcc_mean_vec_kurtosis': 0
    }
    for i in range(1, 5): feat[f'f{i}_mean'] = 0
    for i in range(N_MFCC):
        feat[f'mfcc_mean_t_{i}'] = 0
        feat[f'mfcc_var_t_{i}'] = 0
        feat[f'mfcc_skew_t_{i}'] = 0
        feat[f'mfcc_kurt_t_{i}'] = 0
    return feat


# ==========================================
# 3. 核心算法：声学特征提取引擎
# ==========================================
def extract_acoustic_features(audio_path):
    features = get_empty_acoustic_features()
    try:
        # ---------------------------------------------------------
        # Part 1: Parselmouth (Praat) 物理声学特征
        # ---------------------------------------------------------
        snd = parselmouth.Sound(str(audio_path))

        # 1. F0 / Pitch
        pitch = praat_call(snd, "To Pitch", 0.0, 75, 500)
        pitch_values = pitch.selected_array['frequency']
        voiced_pitch = pitch_values[pitch_values > 0]

        if len(voiced_pitch) > 0:
            features['f0_mean'] = np.mean(voiced_pitch)
            features['f0_std'] = np.std(voiced_pitch)
            features['f0_max'] = np.max(voiced_pitch)
            features['f0_min'] = np.min(voiced_pitch)

            # Jitter & Shimmer
            point_process = praat_call(pitch, "To PointProcess")
            features['jitter_local'] = praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['shimmer_local'] = praat_call((snd, point_process), "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,
                                                   1.6)

        # 2. HNR & Intensity
        harmonicity = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = praat_call(harmonicity, "Get mean", 0, 0)
        features['hnr_mean'] = hnr if not np.isnan(hnr) else 0

        intensity = praat_call(snd, "To Intensity", 100, 0.0)
        int_mean = praat_call(intensity, "Get mean", 0, 0, "energy")
        features['intensity_mean'] = int_mean if not np.isnan(int_mean) else 0

        # 3. 共振峰 Formants
        formant = praat_call(snd, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50)
        for i in range(1, 5):
            f_mean = praat_call(formant, "Get mean", i, 0, 0, "hertz")
            features[f'f{i}_mean'] = f_mean if not np.isnan(f_mean) else 0

        # 4. 停顿分析 Pauses
        textgrid = praat_call(snd, "To TextGrid (silences)", 100, 0, -25, 0.3, 0.1, "silent", "sounding")
        total_duration = praat_call(snd, "Get total duration")

        speaking_duration = 0
        try:
            sounding_intervals = praat_call(textgrid, "Get number of intervals", 2)
            for i in range(1, sounding_intervals + 1):
                speaking_duration += praat_call(textgrid, "Get duration of interval", 2, i)
        except:
            pass

        n_pauses = praat_call(textgrid, "Get number of intervals", 1)
        pause_duration = total_duration - speaking_duration

        features['pause_duration'] = pause_duration
        features['n_pauses'] = n_pauses
        features['pause_rate'] = n_pauses / total_duration if total_duration > 0 else 0
        features['avg_pause_duration'] = pause_duration / n_pauses if n_pauses > 0 else 0

        # ---------------------------------------------------------
        # Part 2: Librosa MFCC 统计特征
        # ---------------------------------------------------------
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) >= MIN_SAMPLES_FOR_FFT:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

            mfcc_mean_t = np.mean(mfccs, axis=1)
            mfcc_var_t = np.var(mfccs, axis=1)
            mfcc_skew_t = skew(mfccs, axis=1, nan_policy='omit')
            mfcc_kurt_t = kurtosis(mfccs, axis=1, nan_policy='omit')

            for i in range(N_MFCC):
                features[f'mfcc_mean_t_{i}'] = mfcc_mean_t[i]
                features[f'mfcc_var_t_{i}'] = mfcc_var_t[i]
                features[f'mfcc_skew_t_{i}'] = mfcc_skew_t[i]
                features[f'mfcc_kurt_t_{i}'] = mfcc_kurt_t[i]

            features['mfcc_mean_vec_skew'] = skew(mfcc_mean_t, nan_policy='omit')
            features['mfcc_mean_vec_kurtosis'] = kurtosis(mfcc_mean_t, nan_policy='omit')

    except Exception as e:
        pass  # 如果遇到任何声学提取引擎崩溃，直接跳过，返回默认的全 0 字典

    return features


# ==========================================
# 4. 主流程：扫描与固化
# ==========================================
def run_audio_handcrafted_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        master_df = pd.read_csv(CSV_PATH)
        print(f"\n>>> [阶段 2-B (Track 2)] 成功加载官方主清单，共计 {len(master_df)} 个任务。")
    except Exception as e:
        print(f"[致命错误] 无法读取清单: {e}");
        return

    extracted_records = []

    # 工业级进度条
    pbar = tqdm(total=len(master_df), desc=" 临床声学特征提取中", unit="audio", ncols=100)

    for index, row in master_df.iterrows():
        wav_path = find_wav_robustly(row['Task_Dir'])

        # 基础元数据 (完美对齐)
        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': row['Task_ID'],
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if wav_path:
            acou_features = extract_acoustic_features(wav_path)
            record.update(acou_features)
        else:
            # 找不到文件补全0
            record.update(get_empty_acoustic_features())

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    # 转换为 DataFrame 并填充可能遗漏的 NaN 为 0
    df_features = pd.DataFrame(extracted_records).fillna(0)
    df_features.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n[大功告成] 音频专家手工特征库提取完毕！")
    print(f"  -> 共生成了 {len(df_features)} 行，每行包含 180+ 维声学特征。")
    print(f"  -> 结构化数据已安全保存至: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_audio_handcrafted_extraction()