import os
import subprocess
import pandas as pd
import numpy as np
import warnings
import shutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心路径配置
# ==========================================
CSV_PATH = r"D:\Code\Project\Dataset\Official_Master_Split.csv"
OUTPUT_DIR = r"D:\Code\Project\Dataset\Offline_Features\Handcrafted_CSV"
# 最终生成的结构化面部肌肉与动力学特征表
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "Video_Facial_Features.csv")

# OpenFace 执行文件绝对路径
OPENFACE_EXE = r"D:\SoftWare\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
# 临时文件存放目录 (用于存放 OpenFace 的逐帧输出，提取完即刻删除)
TEMP_OUT_DIR = os.path.join(OUTPUT_DIR, "OpenFace_Temp")

# 我们关心的核心面部动作单元 (AUs - 强度 Intensity)
# 例如: AU01(内侧眉毛抬起), AU04(皱眉), AU12(微笑/嘴角拉伸)
TARGET_AUS = [f'AU{str(i).zfill(2)}_r' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
# 视线与头部姿态
TARGET_GAZE_POSE = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz']


# ==========================================
# 2. 辅助函数：精准定位 120帧 图片文件夹
# ==========================================
def find_frames_dir(task_dir):
    """复用之前的逻辑：精准定位含有 '120' 的帧文件夹"""
    if not os.path.exists(task_dir): return None
    for sub_item in os.listdir(task_dir):
        sub_path = os.path.join(task_dir, sub_item)
        if os.path.isdir(sub_path) and '120' in sub_item:
            return sub_path
    return None


def get_empty_facial_features():
    """当视频损坏或OpenFace识别失败时，返回全0字典"""
    feat = {}
    for col in TARGET_AUS + TARGET_GAZE_POSE:
        feat[f'{col}_mean'] = 0.0
        feat[f'{col}_std'] = 0.0
    return feat


# ==========================================
# 3. 核心算法：调用 OpenFace 与 特征聚合
# ==========================================
def extract_openface_features(frames_dir):
    """
    后台调用 OpenFace 处理图像序列，并计算均值与方差。
    """
    # 确保临时目录干净
    if os.path.exists(TEMP_OUT_DIR):
        shutil.rmtree(TEMP_OUT_DIR)
    os.makedirs(TEMP_OUT_DIR, exist_ok=True)

    # 构建 OpenFace 后台调用命令
    # -fdir 传入图像序列文件夹
    # -out_dir 指定输出目录
    # -aus -pose -gaze 强制提取表情、姿态、视线
    # -q 安静模式，不弹出视频预览框
    cmd = [
        OPENFACE_EXE,
        '-fdir', frames_dir,
        '-out_dir', TEMP_OUT_DIR,
        '-aus', '-pose', '-gaze', '-q'
    ]

    try:
        # 执行命令，丢弃繁杂的控制台输出
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 寻找生成的 CSV 文件
        csv_files = [f for f in os.listdir(TEMP_OUT_DIR) if f.endswith('.csv')]
        if not csv_files:
            return get_empty_facial_features()

        of_csv_path = os.path.join(TEMP_OUT_DIR, csv_files[0])
        df_frame = pd.read_csv(of_csv_path)

        # 清洗列名 (OpenFace 的列名通常带有前导空格)
        df_frame.columns = df_frame.columns.str.strip()

        # 过滤掉人脸识别失败的帧 (success == 1 代表识别成功)
        if 'success' in df_frame.columns:
            df_frame = df_frame[df_frame['success'] == 1]

        if len(df_frame) == 0:
            return get_empty_facial_features()

        # 开始统计聚合 (Aggregation)
        feat = {}
        for col in TARGET_AUS + TARGET_GAZE_POSE:
            if col in df_frame.columns:
                feat[f'{col}_mean'] = df_frame[col].mean()
                feat[f'{col}_std'] = df_frame[col].std()
            else:
                feat[f'{col}_mean'] = 0.0
                feat[f'{col}_std'] = 0.0

        # 补 0 防止某列全是 NaN 的情况 (如方差计算在只有1帧时)
        for k in feat.keys():
            if pd.isna(feat[k]): feat[k] = 0.0

        return feat

    except Exception as e:
        return get_empty_facial_features()
    finally:
        # 极致硬盘保护：无论成功失败，立刻清理生成的巨量逐帧文件
        if os.path.exists(TEMP_OUT_DIR):
            shutil.rmtree(TEMP_OUT_DIR, ignore_errors=True)


# ==========================================
# 4. 主流程：扫描、分析与固化
# ==========================================
def run_video_handcrafted_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查 OpenFace 引擎是否就位
    if not os.path.exists(OPENFACE_EXE):
        print(f"\n[致命错误] 找不到 OpenFace 引擎: {OPENFACE_EXE}")
        print("请检查路径是否正确，或者是否已正确解压！")
        return

    try:
        master_df = pd.read_csv(CSV_PATH)
        print(f"\n>>> [阶段 2-C (Track 2)] 成功加载官方主清单，共计 {len(master_df)} 个任务。")
    except Exception as e:
        print(f"[致命错误] 无法读取清单: {e}");
        return

    extracted_records = []

    pbar = tqdm(total=len(master_df), desc=" 面部动力学特征提取中", unit="video", ncols=100)

    for index, row in master_df.iterrows():
        frames_dir = find_frames_dir(row['Task_Dir'])

        record = {
            'Subject_ID': row['Subject_ID'],
            'Task_ID': row['Task_ID'],
            'Label_Str': row['Label_Str'],
            'Label_Idx': row['Label_Idx'],
            'Split': row['Split']
        }

        if frames_dir:
            facial_features = extract_openface_features(frames_dir)
            record.update(facial_features)
        else:
            record.update(get_empty_facial_features())

        extracted_records.append(record)
        pbar.update(1)

    pbar.close()

    df_features = pd.DataFrame(extracted_records).fillna(0)
    df_features.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n[大功告成] 视频面部动力学专家特征提取完毕！")
    print(f"  -> 共生成了 {len(df_features)} 行，每行包含极其丰富的表情/视线/姿态均值与方差。")
    print(f"  -> 结构化数据已安全保存至: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_video_handcrafted_extraction()