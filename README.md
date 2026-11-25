
# WDBC – Breast Cancer Classification Benchmark

A comparative machine learning benchmark for binary breast cancer
classification using the Breast Cancer Wisconsin (Diagnostic) Dataset.

The objective of this project is to evaluate and compare five classical
machine learning algorithms in terms of classification performance
and computational efficiency.

## Overview

This project focuses on the classification of breast tumors into two
categories:

- Benign (B)
- Malignant (M)

All models are trained and evaluated using the same dataset, data split,
and evaluation metrics to ensure a consistent comparison.

## Dataset

The Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC) is publicly
available through the UCI Machine Learning Repository.

- 569 samples
- 30 numerical input features
- 357 benign samples
- 212 malignant samples
- Binary classification task

The features describe characteristics of cell nuclei extracted from
digitized images of fine needle aspirates (FNA) of breast masses.

**Dataset source:**

[UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Machine Learning Models

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler, max_iter=1000 |
| Gaussian Naive Bayes | Default parameters |
| k-Nearest Neighbors (k-NN) | StandardScaler, k=5 |
| Support Vector Machine (SVM) | StandardScaler, RBF kernel |
| Random Forest | 100 decision trees, parallel execution |

## Methodology

The experimental workflow consists of the following steps:

1. Load and preprocess the dataset.
2. Remove the sample ID from the input features.
3. Encode the diagnosis as a binary target variable.
4. Perform a stratified 80/20 train-test split.
5. Apply feature standardization where required.
6. Train all five machine learning models.
7. Evaluate classification performance on the test dataset.
8. Measure training and prediction times.
9. Visualize and compare the results.

### Experimental Configuration

| Parameter | Value |
|---|---|
| Training dataset | 80% |
| Test dataset | 20% |
| Random seed | 42 |
| Stratified sampling | Enabled |
| Benign class | 0 |
| Malignant class | 1 |
| Feature scaling | StandardScaler |

Feature scaling is integrated into scikit-learn pipelines for
Logistic Regression, k-NN, and SVM to prevent data leakage during
preprocessing.

## Evaluation Metrics

The classification performance is evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified samples |
| Precision | Proportion of predicted malignant samples that are actually malignant |
| Recall (Sensitivity) | Proportion of malignant samples correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between benign and malignant samples |
| Confusion Matrix | Distribution of correct and incorrect predictions |
| Training Time | Computational time required to fit the model |
| Prediction Time | Computational time required to classify the test dataset |

Precision, recall, and F1-score are calculated with malignant tumors
as the positive class.

Recall is particularly relevant in this context, as false-negative
predictions correspond to malignant tumors incorrectly classified
as benign.

## Results and Model Comparison

### Confusion Matrices

Comparison of true positives, true negatives, false positives,
and false negatives for each classification model.

![Confusion Matrices](https://github.com/user-attachments/assets/cd2fd47c-60bd-4b1e-a9fd-efef7fbf483c)

### Classification Performance

Comparison of accuracy, precision, recall, and F1-score across
all evaluated machine learning models.

![Classification Performance](https://github.com/user-attachments/assets/c8bac773-e6d7-4002-ae48-199735d7ef8f)

### ROC Curve and AUC Comparison

Comparison of the true positive rate and false positive rate
across different decision thresholds.

The area under the ROC curve (AUC) provides a threshold-independent
measure of classification performance.

![ROC Curve Comparison](https://github.com/user-attachments/assets/ffaa6e4f-4a08-455d-9bea-d70ca3fa1253)

### Computational Performance

Comparison of model training and prediction times measured
using Python's high-resolution performance counter.

![Computational Performance](https://github.com/user-attachments/assets/850a5c14-4ea2-4375-907b-c24affd81fd6)

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12.x |
| scikit-learn | 1.4.1.post1 |
| pandas | 2.1.4 |
| NumPy | 1.26.4 |
| Matplotlib | 3.6.3 |

### Installation

Install the required Python dependencies:

```bash
pip install scikit-learn==1.4.1.post1 pandas==2.1.4 numpy==1.26.4 matplotlib==3.6.3
```

### Usage

1. Download the WDBC dataset from the UCI Machine Learning Repository.
2. Place the `wdbc.data` file in the working directory.
3. Execute the Python benchmark script.

The script generates:

- Classification metrics for each model
- Individual confusion matrices
- ROC curves including AUC scores
- Classification performance comparison
- Training and prediction time comparison

## Reproducibility and Limitations

A fixed random seed and a stratified train-test split are used
to improve the reproducibility of the experiments.

However, the benchmark is based on a single train-test split
without systematic hyperparameter optimization or cross-validation.

Computational performance may vary depending on hardware,
operating system, and background processes.

Potential extensions include:

- Stratified k-fold cross-validation
- Hyperparameter optimization
- Statistical significance testing
- Repeated runtime measurements
- Feature importance analysis
- Evaluation on independent external datasets

## Disclaimer

This project is intended for educational and research purposes only.
The trained models are not clinically validated and must not be used
as a substitute for professional medical diagnosis.

## References

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995).
Breast Cancer Wisconsin (Diagnostic).
UCI Machine Learning Repository.

https://doi.org/10.24432/C5DW2B
