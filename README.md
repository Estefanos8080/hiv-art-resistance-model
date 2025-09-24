# HIV Drug Resistance Prediction for NRTI (3TC) using CNN

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for predicting HIV drug resistance to the NRTI drug **3TC (TTC)**. The model is trained on sequence data and uses 5-fold cross-validation for evaluation. This code is a refactoring of the original R code from the paper **"Deep learning predicts HIV drug resistance from viral sequences"** ([PMC7290575](https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/)), adapted to focus on the ART drug **3TC**.

---

## Table of Contents
1. [Overview](#overview)
2. [Code Logic](#code-logic)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architecture](#model-architecture)
   - [Cross-Validation](#cross-validation)
   - [Performance Metrics](#performance-metrics)
   - [Saving Results](#saving-results)
3. [Purpose and Context](#purpose-and-context)
4. [Usage](#usage)
5. [Output Files](#output-files)
6. [Dependencies](#dependencies)
7. [Assumptions](#assumptions)
8. [References](#references)

---

## Overview

The goal of this project is to predict HIV drug resistance to the NRTI drug **3TC (TTC)** using sequence data. The model is a 1D CNN that takes as input the amino acid sequence of the HIV Reverse Transcriptase (RT) gene and outputs a binary prediction:
- `1`: Resistant to 3TC.
- `0`: Not resistant to 3TC.

This implementation is a refactoring of the original R code from the paper **"Deep learning predicts HIV drug resistance from viral sequences"** ([PMC7290575](https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/)). The original study used deep learning models to predict resistance to multiple ART drugs, but this implementation focuses specifically on **3TC**.

---

## Code Logic

### Data Preprocessing
1. **Input Data**:
   - The input is a FASTA file (`ttc.fasta`) containing sequences and their corresponding resistance labels.
   - Each sequence is labeled as `1` (resistant) or `0` (not resistant).

2. **Sequence Encoding**:
   - Sequences are converted into numerical format using a dictionary mapping:
     ```
     A -> 1, B -> 2, ..., Z -> 26, . -> 27, # -> 28, ~ -> 29, * -> 30
     ```
   - Sequences are padded to a fixed length of 240 (the length of the RT gene).

3. **Class Weights**:
   - Class weights are calculated to handle imbalanced data (more resistant or non-resistant samples).

### Model Architecture
The CNN model consists of the following layers:
1. **Embedding Layer**:
   - Converts integer-encoded sequences into dense vectors of fixed size (`output_dim=128`).

2. **1D Convolutional Layer**:
   - Applies 32 filters with a kernel size of 9 and ReLU activation.

3. **Max Pooling Layer**:
   - Reduces the sequence length by taking the maximum value over a window of size 5.

4. **Second 1D Convolutional Layer**:
   - Applies another 32 filters with a kernel size of 9 and ReLU activation.

5. **Flatten Layer**:
   - Flattens the output for the dense layer.

6. **Dense Layer**:
   - A single neuron with a sigmoid activation function for binary classification.

The model is compiled using:
- **Optimizer**: RMSprop.
- **Loss Function**: Binary cross-entropy.
- **Metric**: Accuracy.

### Cross-Validation
- The dataset is split into 5 folds using `KFold` from `sklearn.model_selection`.
- For each fold:
  - The model is trained on 4 folds and validated on the remaining fold.
  - Performance metrics (accuracy, TP, FP, TN, FN) are calculated.

### Performance Metrics
- **True Positives (TP)**: Resistant sequences correctly predicted as resistant.
- **False Positives (FP)**: Non-resistant sequences incorrectly predicted as resistant.
- **True Negatives (TN)**: Non-resistant sequences correctly predicted as non-resistant.
- **False Negatives (FN)**: Resistant sequences incorrectly predicted as non-resistant.

### Saving Results
- **Evaluation Results**:
  - Saved in `ttc.fasta.eval_results.csv` with columns: `fold`, `accuracy`, `loss`, `TP`, `FP`, `TN`, `FN`.
- **ROC Outputs**:
  - Saved in `ttc.fasta.roc.fold_X.csv` with columns: `Predicted_Probability`, `Predicted_Class`, `True_Label`.

---

## Purpose and Context

### Background
The original study ([PMC7290575](https://pmc.ncbi.nlm.nih.gov/articles/PMC7290575/)) demonstrated the effectiveness of deep learning models in predicting HIV drug resistance from viral sequences. The study highlighted the following key points:
1. **Importance of Drug Resistance Prediction**:
   - HIV drug resistance is a major challenge in antiretroviral therapy (ART).
   - Accurate prediction of resistance can help clinicians choose effective drug regimens and improve patient outcomes.

2. **Role of Deep Learning**:
   - Deep learning models, such as CNNs, can capture complex patterns in viral sequences that are associated with drug resistance.
   - These models outperform traditional methods like logistic regression and decision trees.

3. **Focus on Multiple Drugs**:
   - The original study trained models for multiple ART drugs, including NNRTIs, NRTIs, and PIs.
   - This implementation focuses specifically on **3TC**, an NRTI drug, to demonstrate the applicability of the approach to individual drugs.

### Refactoring and Adaptation
- This implementation refactors the original R code into Python, making it more accessible to the Python ecosystem.
- The focus on **3TC** allows for a detailed exploration of the model's performance on a single drug, which can be extended to other drugs in future work.
