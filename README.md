# Ai-Based-Lyric-Intelligibility-Detector-# Ai sem Project
# ICASSP 2026 CADENZA Challenge: Predicting Lyric Intelligibility

## Project Overview

This project was developed for the AI project and ICASSP 2026 Cadenza Challenge. It addresses a core regression task: **predicting lyric intelligibility** (Word Correct Rate, or WCR) in song excerpts, which is crucial for understanding how well listeners perceive lyrics under various conditions.

The work was structured into three phases:
1.  **Problem and Data Understanding:** Extensive Exploratory Data Analysis (EDA) on audio and text features (e.g., Duration, Token Length, Spectrograms).
2.  **Baseline Pipeline Implementation:** Development of the full end-to-end ML pipeline and comparison of three single-feature baselines (Sigmoid, Ridge, Polynomial).
3.  **Proposed Solution and Enhancement:** Implementation of a non-linear feature engineering technique and a regularized ensemble model to significantly improve performance.

## Pipeline & Baseline Implementation

The initial work focused on building a robust, reusable pipeline capable of handling the dataset's metadata and precomputed STOI scores.

* **Pipeline Components:** Included dedicated scripts for `data_loader.py` and `preprocess.py`, ensuring a reproducible data split (80/20 train/test).
* **Baseline Selection:** Three models were tested against a single input feature (STOI score). **Baseline B (Regularized Ridge Regression)** was selected for further improvement due to its strong performance balance.
    * *Initial Performance (Validation RMSE):* 35.33

## Proposed Solution: Multivariate Polynomial Ensemble

The core enhancement was an advanced feature engineering and modeling approach to overcome the limitations of the single-feature baseline:

1.  **Multivariate Feature Generation:** The feature set was expanded from one to three, including `stoi` (perceptual meter), `n_words` (lyric density), and `hearing_loss` (listener ability).
2.  **Non-Linearity:** Used `sklearn.preprocessing.PolynomialFeatures(degree=2)` to generate a **9-feature set** that captures complex, non-linear interactions (e.g., $stoi^2$, $stoi \times hearing\_loss$).
3.  **Regularized Ensemble:** The enhanced feature set was fed into a **Weighted Ensemble** of Ridge, Lasso, and ElasticNet models, which provided a robust final prediction.

## Results and Quantified Impact

The proposed multivariate approach successfully validated the hypothesis that external features are essential for predicting lyric intelligibility, yielding a significant performance gain over the baseline.

| Metric | Baseline B | Proposed Solution | Improvement |
| :--- | :--- | :--- | :--- |
| **Validation RMSE** | 35.33 | **33.70** | **4.62% Reduction** |
| **Validation R² Score** | 0.0417 | **0.1281** | **Tripled variance explanation** |

