"""
Load the trained model.
Receive transaction data in JSON.
Apply the same preprocessing used during training.
Generate a fraud probability.
Apply the selected classification threshold.
Return the result as JSON.
"""

from pathlib import Path
import json
import joblib
import pandas as pd


# 1. Define paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
INPUT_PATH = ROOT / "input.json"


# 2. Load trained model
model = joblib.load(MODEL_PATH)


# 3. Load input JSON
with open(INPUT_PATH, "r") as file:
    data = json.load(file)

input_data = pd.DataFrame([data])


# 4. Predict probability
probability = model.predict_proba(input_data)[0, 1]


# 5. Set classification threshold
threshold = 0.3


# 6. Make classification
if probability >= threshold:
    class_id = 1
    prediction = "Fraud"
else:
    class_id = 0
    prediction = "Legitimate"


# 7. Create output
output = {
    "prediction": prediction,
    "class_id": class_id,
    "probability": probability,
    "threshold": threshold,
    "status": "success"
}


# 8. Return output as JSON
print(json.dumps(output, indent=4))