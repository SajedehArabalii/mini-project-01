# Credit Card Fraud Detection Pipeline

## 1. Project Overview

This project implements an end-to-end machine learning pipeline for credit card fraud detection. The goal is to classify credit card transactions as legitimate or fraudulent.

The pipeline covers data preparation, train/test splitting, feature scaling, model training, cross-validation, hyperparameter experiments, threshold selection, model evaluation, and prediction on new transactions.

---

## 2. Problem Description

### Business Scenario

Credit card fraud is a major challenge for financial institutions. The goal of this project is to build a machine learning system that can identify potentially fraudulent credit card transactions.

### Objective

The objective is to build an end-to-end machine learning pipeline that classifies transactions as:

- `0` → Legitimate
- `1` → Fraudulent

The project focuses on reliable fraud detection rather than accuracy alone.

---

## 3. Dataset

### Dataset Information

- Dataset: Credit Card Fraud Detection Dataset
- Source: Kaggle
- Samples: 283,726
- Features: 30
- Fraudulent transactions: 473
- Legitimate transactions: 283,253
- Fraud ratio: 0.167%.

### Features

The dataset contains:

- `Time`
- `V1`–`V28`
- `Amount`
- `Class` — target variable

The `V1`–`V28` features are anonymized numerical features.

### Class Imbalance

Fraudulent transactions represent only about 0.167% of the dataset. Because legitimate transactions greatly outnumber fraudulent transactions, a model could achieve very high Accuracy while detecting very few fraudulent transactions. Therefore, Precision, Recall, and F1-score are more informative for evaluating fraud detection performance.

---

## 4. Project Structure

```text
mini-project-01/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── data_prep.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
└── reports/
    └── experiments.md