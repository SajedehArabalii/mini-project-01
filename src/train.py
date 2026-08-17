"""
The following models are compared:
    Logistic Regression
    K-Nearest Neighbors (KNN)
    Decision Tree
Experiments include:
    5-Fold Stratified Cross-Validation
    Feature scaling
    Hyperparameter tuning
    Classification threshold selection
Where to use the scaled data:
    Logistic Regression → scaled data
    KNN                 → scaled data
    Decision Tree       → original data
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from data_prep import prepare_data


"""
load prepared data
"""
(
    X_train,
    X_test,
    y_train,
    y_test,
) = prepare_data()

"""
1- Logistic Regression
    Pipeline + Scaling
    5-fold CV
    Hyperparameter tuning
"""

# Pipeline + Scaling
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# 5-fold CV
skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=2
)

cv_results = cross_validate(
    logreg_pipeline,
    X_train,
    y_train, 
    cv=skf, 
    scoring = ["accuracy", "precision", "recall", "f1"]
)

# print("Logistic Regression - 5-Fold CV")
# print("Accuracy:", cv_results["test_accuracy"].mean())
# print("Precision:", cv_results["test_precision"].mean())
# print("Recall:", cv_results["test_recall"].mean())
# print("F1:", cv_results["test_f1"].mean())

# Hyperparameter tuning
param_grid = {
    "logreg__C" : [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    logreg_pipeline,
    param_grid,
    cv=skf,
    scoring="f1"
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("Best CV F1:")
print(grid.best_score_)
"""
2- KNN
    Pipeline + Scaling
    5-fold CV
    Hyperparameter tuning
"""
# Pipeline + scaling
knn_pipeline = Pipeline([
    ("scaler" , StandardScaler()),
    ("knn" , KNeighborsClassifier())
])

# 5-fold  CV
knn_cv_results = cross_validate(
    knn_pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=["accuracy", "precision", "recall", "f1"]
)
# print("\nKNN - 5-Fold CV")
# print("Accuracy:", knn_cv_results["test_accuracy"].mean())
# print("Precision:", knn_cv_results["test_precision"].mean())
# print("Recall:", knn_cv_results["test_recall"].mean())
# print("F1:", knn_cv_results["test_f1"].mean())

# Hyperparameter Tuning
knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 11, 15]
}

knn_grid = GridSearchCV(
    knn_pipeline,
    knn_param_grid,
    cv=skf,
    scoring={
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    },
    refit="f1"
)

knn_grid.fit(X_train, y_train)

knn_results = pd.DataFrame(knn_grid.cv_results_)

print("\n=== KNN Hyperparameter Experiment ===")

for i in range(len(knn_results)):
    print(f"\nK = {knn_results.loc[i, 'param_knn__n_neighbors']}")
    print(f"Precision: {knn_results.loc[i, 'mean_test_precision']:.3f}")
    print(f"Recall:    {knn_results.loc[i, 'mean_test_recall']:.3f}")
    print(f"F1:        {knn_results.loc[i, 'mean_test_f1']:.3f}")

print("\nBest K:")
print(knn_grid.best_params_)

print("Best CV F1:")
print(knn_grid.best_score_)

print("\nKNN - Best Parameters:")
print(knn_grid.best_params_)

print("KNN - Best CV F1:")
print(knn_grid.best_score_)


"""
2.5 experiment, KNN without scaling and comparison
"""
knn_unscaled = KNeighborsClassifier()

knn_unscaled_cv_results = cross_validate(
    knn_unscaled,
    X_train,
    y_train,
    cv=skf,
    scoring=['precision', 'recall', 'f1']
)

print("\n=== KNN Scaling Experiment ===")

print("Without Scaling:")
print("Precision:", knn_unscaled_cv_results["test_precision"].mean())
print("Recall:", knn_unscaled_cv_results["test_recall"].mean())
print("F1:", knn_unscaled_cv_results["test_f1"].mean())

print("\nWith Scaling:")
print("Precision:", knn_cv_results["test_precision"].mean())
print("Recall:", knn_cv_results["test_recall"].mean())
print("F1:", knn_cv_results["test_f1"].mean())

"""
3- Decision Tree
    5-fold CV
    Hyperparameter tuning
"""
# 5_fold CV
tree = DecisionTreeClassifier(random_state=2)
tree_cv_results = cross_validate(
    tree,
    X_train,
    y_train,
    cv = skf,
    scoring = ["accuracy", "precision", "recall", "f1"]
)
# print("\nDecision Tree - 5-Fold CV")
# print("Accuracy:", tree_cv_results["test_accuracy"].mean())
# print("Precision:", tree_cv_results["test_precision"].mean())
# print("Recall:", tree_cv_results["test_recall"].mean())
# print("F1:", tree_cv_results["test_f1"].mean())

#Hyperparameter tuning

tree_param_grid = {
"max_depth": [3, 5, 10, 15, 20, None]
}
tree_grid = GridSearchCV(
    tree,
    tree_param_grid,
    cv=skf,
    scoring={
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    },
    refit="f1"
)
tree_grid.fit(X_train, y_train)


tree_results = pd.DataFrame(tree_grid.cv_results_)

print("\n=== Decision Tree Hyperparameter Experiment ===")

for i in range(len(tree_results)):
    print(f"\nMax Depth = {tree_results.loc[i, 'param_max_depth']}")
    print(f"Precision: {tree_results.loc[i, 'mean_test_precision']:.3f}")
    print(f"Recall:    {tree_results.loc[i, 'mean_test_recall']:.3f}")
    print(f"F1:        {tree_results.loc[i, 'mean_test_f1']:.3f}")


print("\nDecision Tree - Best Parameters:")
print(tree_grid.best_params_)

print("Decision Tree - Best CV F1:")
print(tree_grid.best_score_)
"""
4- Compare the three
    CV Precision
    CV Recall
    CV F1
"""
# Compare CV Precision
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree"
    ],
    "Accuracy": [
        cv_results["test_accuracy"].mean(),
        knn_cv_results["test_accuracy"].mean(),
        tree_cv_results["test_accuracy"].mean()
    ],
    "Precision": [
        cv_results["test_precision"].mean(),
        knn_cv_results["test_precision"].mean(),
        tree_cv_results["test_precision"].mean()
    ],
    "Recall": [
        cv_results["test_recall"].mean(),
        knn_cv_results["test_recall"].mean(),
        tree_cv_results["test_recall"].mean()
    ],
    "F1": [
        cv_results["test_f1"].mean(),
        knn_cv_results["test_f1"].mean(),
        tree_cv_results["test_f1"].mean()
    ]
})

print("\n=== Model Comparison ===")
print(comparison)

"""
5- Threshold selection
"""
best_knn = knn_grid.best_estimator_
best_knn.fit(X_train, y_train)

y_prob = best_knn.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]

print("\n=== KNN Threshold Comparison ===")

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

"""
6- Final test
    which model
    which hyperparameters
    which threshold
giving the final
    accuracy
    precision
    Recall
    F!
    confusion matrix
"""
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

final_model = knn_grid.best_estimator_
final_model.fit(X_train, y_train)

# Save the complete model pipeline
joblib.dump(final_model, MODEL_PATH)

# Save the fitted scaler separately
joblib.dump(final_model.named_steps["scaler"], SCALER_PATH)

y_prob = final_model.predict_proba(X_test)[:, 1]
final_threshold = 0.3
y_pred = (y_prob >= final_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n=== Final Model Evaluation ===")

print("Model: KNN")
print("Threshold:", final_threshold)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)