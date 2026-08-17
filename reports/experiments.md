# Experiments Report

## 1. Experimental Setup

### Dataset

- Dataset: 
- Number of samples:
- Number of features:
- Fraudulent transactions:
- Legitimate transactions:
- Fraud ratio:
- Missing values:
- Duplicates:

### Train/Test Split

- Test size:
- Stratification:
- Random state:

### Evaluation Metrics

The following metrics were used:

- Accuracy:
- Precision:
- Recall:
- F1-score:

---

## 2. Initial Hypothesis

### Logistic Regression

Hypothesis:

### KNN

Hypothesis:

### Decision Tree

Hypothesis:

### Expected Important Metric

Hypothesis:

---

# 3. Experiment 1 — Model Comparison

## Objective

Compare Logistic Regression, KNN, and Decision Tree using 5-Fold Stratified Cross-Validation.

## Method

- Cross-validation:
- Number of folds:
- Scoring metrics:

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | | | | |
| KNN | | | | |
| Decision Tree | | | | |

## Analysis

- Best model:
- Best metric:
- Worst model:
- What did the results show?
- Was the initial hypothesis correct?

---

# 4. Experiment 2 — Effect of Feature Scaling

## Objective

Determine how feature scaling affects KNN performance.

## Method

Compare KNN:

1. Without scaling
2. With scaling

## Results

| KNN | Precision | Recall | F1 |
|---|---:|---:|---:|
| Without Scaling | | | |
| With Scaling | | | |

## Analysis

- Did scaling improve performance?
- Which metric changed the most?
- Why is KNN affected by scaling?
- What did this experiment teach?

---

# 5. Experiment 3 — KNN Hyperparameter Analysis

## Objective

Investigate how the number of neighbors affects KNN performance.

## Hyperparameters Tested

- K = 3
- K = 5
- K = 7
- K = 11
- K = 15

## Results

| K | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 3 | | | |
| 5 | | | |
| 7 | | | |
| 11 | | | |
| 15 | | | |

## Analysis

- Best K:
- Best F1:
- Best Recall:
- Best Precision:
- Evidence of overfitting:
- Final interpretation:

---

# 6. Experiment 4 — Decision Tree Hyperparameter Analysis

## Objective

Investigate the effect of `max_depth` on Decision Tree performance.

## Hyperparameters Tested

- max_depth = 3
- max_depth = 5
- max_depth = 10
- max_depth = 15
- max_depth = 20
- max_depth = None

## Results

| max_depth | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 3 | | | |
| 5 | | | |
| 10 | | | |
| 15 | | | |
| 20 | | | |
| None | | | |

## Analysis

- Best depth:
- Best F1:
- Evidence of overfitting:
- Effect of increasing tree depth:
- Final interpretation:

---

# 7. Experiment 5 — Classification Threshold

## Objective

Investigate how changing the classification threshold affects fraud detection.

## Thresholds Tested

- 0.3
- 0.5
- 0.7

## Model Used

- Model:
- Hyperparameters:

## Results

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | | | |
| 0.5 | | | |
| 0.7 | | | |

## Analysis

### Recall

What happened when the threshold decreased?

### Precision

What happened when the threshold decreased?

### Trade-off

Explain the relationship between:

- False Positives
- False Negatives
- Precision
- Recall

### Selected Threshold

- Threshold:
- Reason:

---

# 8. Final Model Evaluation

## Selected Model

- Model:
- Hyperparameters:
- Threshold:

## Test Set Results

| Metric | Result |
|---|---:|
| Accuracy | |
| Precision | |
| Recall | |
| F1-score | |

## Confusion Matrix

```text
[[TN, FP]
 [FN, TP]]