# Experiments Report

## 1. Experimental Setup

### Dataset

- Dataset: creditcard.csv
- Number of samples: 283726
- Number of features: 30
- Fraudulent transactions: 473
- Legitimate transactions: 283253
- Fraud ratio: ≈ 0.167%.
- Missing values: 0
- Duplicates: 1081

### Train/Test Split

- Test size: 0.2
- Stratification: True
- Random state: 2

### Evaluation Metrics

The following metrics were used:

- Accuracy: yes (although not useful due to class imbalance)
- Precision: yes
- Recall: yes
- F1-score: yes

---

## 2. Initial Hypothesis

### Logistic Regression

Hypothesis: 
I expect Logistic Regression to provide a strong baseline because this is a binary classification problem. However, we must consider its limitations of capturing complex relationships

### KNN

Hypothesis:
I expect KNN to perform reasonably well, especially with feature scaling, because distance-based learning can identify similar transaction patterns. However, the highly imbalanced dataset may affect its fraud detection performance.

### Decision Tree

Hypothesis:
I expect the Decision Tree to perform well because it can capture nonlinear relationships. However, I expect deeper trees to have a higher risk of overfitting.

### Expected Important Metric

Hypothesis:
I expect Recall to be particularly important because missing a fraudulent transaction (False Negative) is undesirable. 

---

# 3. Experiment 1 — Model Comparison

## Objective

Compare Logistic Regression, KNN, and Decision Tree using 5-Fold Stratified Cross-Validation.

## Method

- Cross-validation: GridSearchCV
- Number of folds: 5
- Scoring metrics: Precision, Recall, F1

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression |0.99|0.85|0.58|0.69|
| KNN |0.99|0.92|0.75|0.83|
| Decision Tree |0.99|0.76|0.74|0.75|

## Analysis

- Best model: KNN
- Best metric: F1
- Worst model: Logistic Regression
- What did the results show? KNN performed the best overall
- Was the initial hypothesis correct? KNN performed better than expected benefitting from scaling

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
| Without Scaling |1.0|0.02|0.05|
| With Scaling |0.92|0.75|0.83|

## Analysis

- Did scaling improve performance? Yes
- Which metric changed the most? Recall, increasing from 0.02 to 0.75
- Why is KNN affected by scaling? Because it uses distances, which does not perform well with imbalanced data, unless it is scaled
- What did this experiment teach? feature scaling is essential for KNN. 

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
| 3 |0.93|0.78|0.85|
| 5 |0.92|0.75|0.83|
| 7 |0.91|0.74|0.81|
| 11 |0.90|0.74|0.81|
| 15 |0.88|0.74|0.80|

## Analysis

- Best K: 3
- Best F1: 0.85 at K=3
- Best Recall: 0.78 at K=3
- Best Precision: 0.93 at K=3
- Evidence of overfitting: No clear evidence
- Final interpretation: K=3 performed best across all metrics

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
| 3 |0.85|0.69|0.76|
| 5 |0.88|0.74|0.80|
| 10 |0.88|0.74|0.80|
| 15 |0.79|0.74|0.76|
| 20 |0.77|0.74|0.75|
| None |0.76|0.74|0.75|

## Analysis

- Best depth: 10
- Best F1: 0.807 at max_depth = 10
- Best Precision: 0.887 at max_depth = 10
- Best Recall: 0.744 at max_depth = 5
- Evidence of overfitting: Yes. As tree depth increases beyond 10, Precision and F1 decrease, suggesting that deeper trees generalize less effectively.
- Effect of increasing tree depth:Performance improved from depth 3 to 10, then generally decreased as depth increased.
- Final interpretation:max_depth = 10 provided the best balance of Precision, Recall, and F1.

---

# 7. Experiment 5 — Classification Threshold

## Objective

Investigate how changing the classification threshold affects fraud detection.

## Thresholds Tested

- 0.3
- 0.5
- 0.7

## Model Used

- Model: KNN
- Hyperparameters: K = 3

## Results

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 |0.66|0.8|0.72|
| 0.5 |0.93|0.76|0.84|
| 0.7 |0.7|0.69|0.80|

## Analysis

### Recall

What happened when the threshold decreased?
When the threshold decreased from 0.5 to 0.3, Recall increased from 0.76 to 0.80.

### Precision

What happened when the threshold decreased?
When the threshold decreased from 0.5 to 0.3, Precision decreased from 0.93 to 0.66

### Trade-off

Explain the relationship between:

Lowering the threshold detects more fraudulent transactions, reducing False Negatives, but increases False Positives and therefore decreases Precision.
Increasing the threshold generally reduces False Positives but can increase False Negatives.

### Selected Threshold

- Threshold: 0.3
- Reason: It provides the highest Recall (0.80), which is important because detecting fraudulent transactions is the primary goal.

---

# 8. Final Model Evaluation

## Selected Model

- Model: KNN
- Hyperparameters: K = 3
- Threshold: 0.3

## Test Set Results

| Metric | Result |
|---|---:|
| Accuracy |0.99|
| Precision |0.66|
| Recall |0.8|
| F1-score |0.72|

## Confusion Matrix

```text
[[56612, 39]
 [19, 76]]