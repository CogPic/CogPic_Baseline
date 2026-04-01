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

## 代码仓库结构详解 (Repository Structure)

本项目共包含 20 个核心执行脚本，按照功能流转逻辑严格划分为四大模块：**离线手工特征提取**、**单模态基准测试**、**跨模态深度融合** 以及 **机器学习融合与可解释性**。

### 模块一：离线手工特征提取 (Offline Feature Extraction)
本模块包含 **3** 个脚本，专门用于利用各种领域内专家工具包，将原始的非结构化数据转化为具备明确临床/语言学意义的结构化高维特征表格（CSV）。
* **`Extract_Audio_Handcrafted_Features.py`**：调用 `parselmouth` (Praat) 和 `librosa`，提取 180+ 维声学特征（F0, 微扰, HNR, 共振峰, 停顿统计, MFCC）。
* **`Extract_Text_Handcrafted_Features.py`**：结合 `Stanza` 与 `jieba`，提取 60-80 维语言学特征（词汇丰富度 MATTR, 句法树高度, 词性比例, 语义跳跃度）。
* **`Extract_Video_Handcrafted_Features.py`**：后台静默调用 **OpenFace**，提取面部动作单元 (AUs)、视线角度 (Gaze) 以及三维头部姿态 (Pose) 的动态时序统计量。

### 模块二：单模态极限性能基准 (Single-Modal Baselines)
本模块包含 **8** 个脚本，用于确立各个模态在孤立状态下的性能天花板。
* **文本 (Text)**：
  * `Single_text_modality_DL.py`：BERT 微调、TextCNN、BiLSTM、Att-BiLSTM 深度基准。
  * `Single_text_modality_ML.py`：BERT `[CLS]` 嵌入向量与手工语言学特征拼接的传统 ML 分类。
* **音频 (Audio)**：
  * `Single_audio_modality_DL.py`：基于 Mel 频谱图的视觉架构（SEResNet50, ResNet18, CRNN, ViT）。
  * `Single_audio_modality_DL_end_to_end.py`：基于原始一维波形的端到端 RawWave-LSTM。
  * `Single_audio_modality_ML.py`：调用 OpenSMILE 提取 88 维专家声学特征的 ML 分类。
* **视频 (Video)**：
  * `Single_video_modality_DL.py`：时空 3D 卷积调优（R3D_18, MC3_18, R2Plus1D, ResNet+LSTM）。
  * `Single_video_modality_DL_C3D.py`：经典 C3D 架构的独立实现与防梯度爆炸优化。
  * `Single_video_modality_ML.py`：基于 OpenFace 提取的 AUs、视线和姿态统计量的 ML 基准。

### 模块三：跨模态深度学习融合 (Cross-Modal DL Fusion)
本模块包含 **7** 个脚本，通过冻结单模态预训练骨干网络，利用 Concat-MLP 进行晚期跨模态融合（Late Fusion）。
* **双模态消融实验**：
  * `text_audio_diffExtra.py`：文本 + 音频表征组合收益评估。
  * `text_video_diffExtra.py` 及 `_better.py`：文本 + 视频（如 TextCNN + MC3_18）最优融合基准。
  * `video_audio_diffExtra.py` 及 `_better.py`：纯非语言信号（音频 + 视频）的联合建模效果。
* **三模态满贯实验**：
  * `text_video_audio_diffExtra.py`：全量架构消融，引入多任务颗粒度分析（Global, Pic 1, Pic 2, Pic 3 指标矩阵）。
  * `text_video_audio_diffExtra_better.py`：最强单模态提取器（TextCNN + SEResNet50 + MC3_18）的终极强强联合与盲测。

### 模块四：机器学习融合与可解释性 (ML Fusion & Interpretability)
本模块包含 **2** 个脚本，解决临床医疗场景中至关重要的模型可信度与可解释性问题。
* **`Interpretability_ML_Fusion.py`**：将 Text、Audio、Video 三大手工特征表合并，评估传统分类器，生成 LaTeX 基准表格。
* **`For_SHAP_plot.py`**：自动选出最优树模型，利用 SHAP 计算特征的全局重要性，并渲染/导出纯学术排版风格（黑白灰、Times New Roman）的高分辨率蜂窝图（Beeswarm Plot）。

---

##  环境依赖 (Dependencies)

* **核心环境**：Python 3.8+, PyTorch, Torchvision, Torchaudio
* **模型与架构**：Hugging Face `transformers` (需本地 `bert-base-chinese`), `timm`
* **音频处理**：`librosa`, `soundfile`, `parselmouth` (Praat), `opensmile`
* **文本处理**：`stanza`, `jieba`
* **视频与图像**：`Pillow`, OpenFace 2.2.0+ (需配置本地 `.exe` 路径)
* **机器学习与评估**：`scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `pandas`, `tqdm`

---
