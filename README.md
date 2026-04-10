# CogPic Dataset

https://cogpic.github.io/

Anonymous datasets may be obtained from the corresponding author upon reasonable request for research purposes, but must be approved and comply with relevant data protection regulations.

## Introduction

CogPic is a multimodal dataset designed to evaluate cognitive function. The dataset consists of subjects performing a picture description task, with synchronized collection of their video, audio, and ASR (Automatic Speech Recognition) text data.
This dataset is primarily used for identifying and classifying cognitive states, including Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Healthy Controls (HC). Through its multimodal data, CogPic supports various cognitive research studies and the training of machine learning models.

## Dataset Overview

The CogPic dataset contains the following core components:

- **Video Data**：Records the visual information of subjects while they perform the picture description task.
- **Audio Data**：Voice recordings of subjects describing the pictures.
- **ASR Text Data**：Automatic Speech Recognition (ASR) transcription text derived from the audio.
- **Labels**：Cognitive status labels for each subject, including AD, MCI, and HC classifications.

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
* Audio:
  * Single_audio_modality_DL.py: Vision-based architectures utilizing Mel spectrograms.
  * Single_audio_modality_DL_end_to_end.py: End-to-end architectures based on raw 1D waveforms.
* Video:
  * Single_video_modality_DL.py: Spatiotemporal 3D convolution tuning (R3D_18, MC3_18, R2Plus1D, ResNet+LSTM).
  * Single_video_modality_DL_C3D.py: Experiments with the classical C3D architecture.

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

## Citation
If you use the CogPic dataset in your research, please cite the following:

[CogPic: A Multimodal Dataset for Early Cognitive Impairment Assessment via Picture Description Tasks](https://arxiv.org/abs/2604.01626)


```bibtex
@article{cogpic2026,
  title={CogPic: A Multimodal Dataset for Early Cognitive Impairment Assessment via Picture Description Tasks},
  author={Wu, Liuyu and Feng, Rui and Li, Jie and Xiang, Wentao and Zhang, Yi and Cao, Yin and Song, Siyang and Gu, Xiao and Li, Jianqing and Wang, Wei},
  journal={arXiv preprint arXiv:2604.01626},
  year={2026}
}
