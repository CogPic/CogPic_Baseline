import os
import argparse
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.stats import skew, kurtosis
import warnings
from tqdm import tqdm

# Ignore standard division by zero or short audio warnings
warnings.filterwarnings('ignore')

N_MFCC = 42
MIN_SAMPLES_FOR_FFT = 2048


# ==========================================
# 1. Helper Functions
# ==========================================
def build_path_mapping(base_dir):
    """Dynamically map Task_ID to .wav paths to avoid hardcoded CSV paths."""
    mapping = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                task_id = os.path.basename(root)
                mapping[task_id] = os.path.join(root, file)
    return mapping


def get_empty_acoustic_features():
    """Return a dictionary of zeros when audio is corrupted or too short."""
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
# 2. Core Algorithm: Feature Extraction Engine
# ==========================================
def extract_acoustic_features(audio_path):
    features = get_empty_acoustic_features()
    try:
        # ---------------------------------------------------------
        # Part 1: Parselmouth (Praat) Physical Acoustic Features
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

        # 3. Formants
        formant = praat_call(snd, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50)
        for i in range(1, 5):
            f_mean = praat_call(formant, "Get mean", i, 0, 0, "hertz")
            features[f'f{i}_mean'] = f_mean if not np.isnan(f_mean) else 0

        # 4. Pauses Analysis
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
        # Part 2: Librosa MFCC Statistical Features
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
        pass  # Skip and return zeros if acoustic engine crashes

    return features


# ==========================================
# 3. Main Execution
# ==========================================
def run_audio_handcrafted_extraction(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "Audio_Acoustic_Features.csv")

    try:
        master_df = pd.read_csv(args.csv_path)
        print(f"\n>>> [Data Loading] Master split loaded successfully. Total tasks: {len(master_df)}.")
    except Exception as e:
        print(f"[Fatal Error] Failed to read CSV: {e}")
        return

    # Build dynamic path mapping
    path_mapping = build_path_mapping(args.data_dir)
    extracted_records = []

    pbar = tqdm(total=len(master_df), desc="Extracting Acoustic Features", unit="audio", ncols=100)

    for _, row in master_df.iterrows():
        task_id = str(row['Task_ID']).strip()
        wav_path = path_mapping.get(task_id)

        # Baseline Metadata
        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': task_id,
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if wav_path:
            acou_features = extract_acoustic_features(wav_path)
            record.update(acou_features)
        else:
            record.update(get_empty_acoustic_features())

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    # Convert to DataFrame and fill potential NaNs
    df_features = pd.DataFrame(extracted_records).fillna(0)
    df_features.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[Success] Expert Handcrafted Audio Features extracted successfully!")
    print(f"  -> Generated {len(df_features)} rows, each containing 180+ acoustic features.")
    print(f"  -> Data safely saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Acoustic Feature Extraction")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for audio dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Handcrafted_CSV',
                        help='Directory to save the generated CSV')

    args = parser.parse_args()
    run_audio_handcrafted_extraction(args)