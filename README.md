# Credit Card Fraud Detection — Preliminary Q&A

### 1. Which model do you expect to perform best for fraud detection? Why?

**Decision Tree:** It can capture nonlinear relationships and interactions between features, which may help it distinguish fraudulent transactions from legitimate ones more effectively.

### 2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

**Recall:** In fraud detection, missing a fraudulent transaction can be costly. Therefore, we generally want to minimize **false negatives** and detect as many fraudulent transactions as possible.

### 3. What do you expect to happen if the model predicts all transactions as legitimate?

It would achieve **very high accuracy** because of the severe class imbalance, but its **precision, recall, and F1-score for the fraud class would be 0** because it would detect no fraudulent transactions.

### 4. Do you expect feature scaling to significantly affect KNN performance?

**Yes.** KNN relies on distances between data points, so features with larger scales can dominate the distance calculation and significantly affect the model's performance.

### 5. Do you expect the Decision Tree to overfit? Why?

**Yes.** A Decision Tree can become very complex and memorize the training data, especially when it grows too deep, which can lead to **overfitting** and poor performance on unseen data.
