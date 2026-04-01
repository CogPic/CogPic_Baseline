# CogPic Dataset

https://cogpic.github.io/

Anonymous datasets may be obtained from the corresponding author upon reasonable request for research purposes, but must be approved and comply with relevant data protection regulations.

## 介绍

CogPic 是一个多模态数据集，旨在评估认知功能。数据集由受试者进行图片描述任务，并同步采集其视频、音频和 ASR 文本数据。该数据集主要用于识别和分类认知状态，如阿尔茨海默病 (AD)、轻度认知障碍 (MCI) 和健康对照 (HC)。通过多模态数据，CogPic 可以支持各类认知研究和机器学习模型的训练。

## 数据集概述

CogPic 数据集包含以下几个核心组成部分：

- **视频数据**：记录了受试者进行图片描述任务时的视觉信息。
- **音频数据**：受试者描述图片时的语音录音。
- **ASR文本数据**：音频的自动语音识别 (ASR) 转录文本。
- **标签**：每个受试者的认知状态标签，包括 AD、MCI 和 HC 分类。

数据集旨在为认知功能评估提供一个跨模态的基准，支持研究人员开展多维度分析。

## Detailed Repository Structure

This project contains 20 core execution scripts, strictly divided into four major modules according to functional flow logic: Handcrafted Feature Extraction, Single-Modal Testing, Cross-Modal Fusion, and Machine Learning Fusion & Interpretability.

### Module 1: Feature Extraction
This module contains 3 scripts dedicated to utilizing various domain-expert toolkits to transform raw, unstructured data into structured, high-dimensional feature tables (CSV) with clear clinical and linguistic significance.
* Extract_Audio_Handcrafted_Features.py: Calls parselmouth (Praat) and librosa to extract 180+ dimensional acoustic features (F0, perturbation, HNR, formants, pause statistics, MFCC).
* Extract_Text_Handcrafted_Features.py: Combines Stanza and jieba to extract 60-80 dimensional linguistic features (vocabulary richness MATTR, syntactic tree height, part-of-speech ratios, semantic leaps).
* Extract_Video_Handcrafted_Features.py: Silently calls OpenFace in the background to extract dynamic temporal statistics of facial Action Units (AUs), Gaze, and 3D Head Pose.

### Module 2: Single-Modal Baselines
This module contains 8 scripts used to establish the performance limits and baselines for each individual modality.
* Text:
  * Single_text_modality_DL.py: Baselines for BERT fine-tuning, TextCNN, BiLSTM, and Att-BiLSTM.
  * Single_text_modality_ML.py: Traditional ML classification concatenating BERT [CLS] embedding vectors with handcrafted linguistic features.
* Audio:
  * Single_audio_modality_DL.py: Vision-based architectures utilizing Mel spectrograms.
  * Single_audio_modality_DL_end_to_end.py: End-to-end architectures based on raw 1D waveforms.
  * Single_audio_modality_ML.py: ML classification calling OpenSMILE to extract 88-dimensional expert acoustic features.
* Video:
  * Single_video_modality_DL.py: Spatiotemporal 3D convolution tuning (R3D_18, MC3_18, R2Plus1D, ResNet+LSTM).
  * Single_video_modality_DL_C3D.py: Experiments with the classical C3D architecture.
  * Single_video_modality_ML.py: ML baselines based on OpenFace-extracted statistics for AUs, gaze, and head pose.

### Module 3: Cross-Modal Fusion
This module contains 7 scripts, utilizing a Concat-MLP for Late Fusion by freezing single-modal pre-trained backbone networks.
* Bi-Modal Ablation Experiments:
  * text_audio_diffExtra.py: Text + Audio fusion.
  * text_video_diffExtra.py and _better.py: Text + Video fusion.
  * video_audio_diffExtra.py and _better.py: Audio + Video fusion.
* Tri-Modal Grand Slam Experiments:
  * text_video_audio_diffExtra.py: Introduces multi-task analysis (Global, Pic 1, Pic 2, and Pic 3 metric matrices).
  * text_video_audio_diffExtra_better.py: Combination of optimal extractors (TextCNN + SEResNet50 + MC3_18).

### Module 4: ML Fusion & Interpretability
This module contains 2 scripts to address the crucial issues of model credibility and interpretability in clinical medical scenarios.
* Interpretability_ML_Fusion.py: Merges the three handcrafted feature tables (Text, Audio, Video) to evaluate traditional classifiers.
* For_SHAP_plot.py: Automatically selects the optimal tree-based model and utilizes SHAP to calculate the global importance of features.

---

## Dependencies

* Core Environment: Python 3.8+, PyTorch, Torchvision, Torchaudio
* Models & Architectures: Hugging Face transformers, timm
* Audio Processing: librosa, soundfile, parselmouth (Praat), opensmile
* Text Processing: stanza, jieba
* Video & Images: Pillow, OpenFace 2.2.0+ 
* Machine Learning & Evaluation: scikit-learn, xgboost, lightgbm, catboost, shap, pandas, tqdm

---
