import os
import subprocess
import argparse
import pandas as pd
import warnings
import shutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Target Action Units (AUs - Intensity)
TARGET_AUS = [f'AU{str(i).zfill(2)}_r' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
# Target Gaze and Head Pose
TARGET_GAZE_POSE = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz']


# ==========================================
# 1. Helper Functions
# ==========================================
def build_path_mapping(base_dir):
    """Dynamically map Task_ID to the folder containing 120 frames."""
    mapping = {}
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if '120' in d:
                task_id = os.path.basename(root)
                mapping[task_id] = os.path.join(root, d)
    return mapping


def get_empty_facial_features():
    """Return dict of zeros if video is missing or OpenFace fails."""
    feat = {}
    for col in TARGET_AUS + TARGET_GAZE_POSE:
        feat[f'{col}_mean'] = 0.0
        feat[f'{col}_std'] = 0.0
    return feat


# ==========================================
# 2. Core Algorithm: OpenFace Execution
# ==========================================
def extract_openface_features(frames_dir, openface_exe, temp_out_dir):
    """
    Calls OpenFace in the background to process the image sequence.
    """
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
    os.makedirs(temp_out_dir, exist_ok=True)

    cmd = [
        openface_exe,
        '-fdir', frames_dir,
        '-out_dir', temp_out_dir,
        '-aus', '-pose', '-gaze', '-q'
    ]

    try:
        # Suppress massive console output from OpenFace
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        csv_files = [f for f in os.listdir(temp_out_dir) if f.endswith('.csv')]
        if not csv_files:
            return get_empty_facial_features()

        of_csv_path = os.path.join(temp_out_dir, csv_files[0])
        df_frame = pd.read_csv(of_csv_path)

        # Clean column names (OpenFace adds leading spaces)
        df_frame.columns = df_frame.columns.str.strip()

        # Filter out frames where face tracking failed
        if 'success' in df_frame.columns:
            df_frame = df_frame[df_frame['success'] == 1]

        if len(df_frame) == 0:
            return get_empty_facial_features()

        # Aggregate temporal features
        feat = {}
        for col in TARGET_AUS + TARGET_GAZE_POSE:
            if col in df_frame.columns:
                feat[f'{col}_mean'] = df_frame[col].mean()
                feat[f'{col}_std'] = df_frame[col].std()
            else:
                feat[f'{col}_mean'] = 0.0
                feat[f'{col}_std'] = 0.0

        for k in feat.keys():
            if pd.isna(feat[k]): feat[k] = 0.0

        return feat

    except Exception as e:
        return get_empty_facial_features()
    finally:
        # Extreme Disk Protection: Clean up massive frame-by-frame outputs immediately
        if os.path.exists(temp_out_dir):
            shutil.rmtree(temp_out_dir, ignore_errors=True)


# ==========================================
# 3. Main Execution
# ==========================================
def run_video_handcrafted_extraction(args):
    os.makedirs(args.output_dir, exist_ok=True)
    temp_out_dir = os.path.join(args.output_dir, "OpenFace_Temp")
    output_csv_path = os.path.join(args.output_dir, "Video_Facial_Features.csv")

    if not os.path.exists(args.openface_exe):
        print(f"\n[Fatal Error] OpenFace executable not found at: {args.openface_exe}")
        print("Please check the path or ensure OpenFace is installed correctly.")
        return

    try:
        master_df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[Fatal Error] Failed to read master CSV: {e}")
        return

    path_mapping = build_path_mapping(args.data_dir)
    extracted_records = []

    pbar = tqdm(total=len(master_df), desc="Extracting Facial Dynamics", unit="video", ncols=100)

    for _, row in master_df.iterrows():
        task_id = str(row['Task_ID']).strip()
        frames_dir = path_mapping.get(task_id)

        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': task_id,
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if frames_dir:
            facial_features = extract_openface_features(frames_dir, args.openface_exe, temp_out_dir)
            record.update(facial_features)
        else:
            record.update(get_empty_facial_features())

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    df_features = pd.DataFrame(extracted_records).fillna(0)
    df_features.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n[Success] Video Facial Dynamics Features extracted successfully!")
    print(f"  -> Generated {len(df_features)} rows containing AU/Gaze/Pose mean and std.")
    print(f"  -> Data safely saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Facial Feature Extraction using OpenFace")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to master split CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory for video frames dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/Handcrafted_CSV',
                        help='Directory to save output CSV')
    parser.add_argument('--openface_exe', type=str, required=True,
                        help='Absolute path to OpenFace FeatureExtraction.exe')

    args = parser.parse_args()
    run_video_handcrafted_extraction(args)