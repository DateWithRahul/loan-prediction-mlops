
import os
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 1. Load the trained model artifact
MODEL_PATH = "./model/loan_default.pkl"
DATA_PATH = "./data/x_test_sample.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please run train.py first!"
    )

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

# 2. Load the test sample dataframe
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Test data sample not found at {DATA_PATH}. Please run train.py first!"
    )

test_df = pd.read_csv(DATA_PATH)

# Safely drop 'Unnamed: 0' only if it exists in the dataframe layout
if "Unnamed: 0" in test_df.columns:
    test_df.drop("Unnamed: 0", axis=1, inplace=True)

# 3. Sample a single record for inference testing
x_test_sample = test_df.sample(1, random_state=42)
print("\n--- Selected Inference Sample ---")
print(x_test_sample)

# 4. CRITICAL MLOPS STEP: Align features to match training schema sequence exactly
x_test_sample = x_test_sample[model.feature_names_in_]

# 5. Execute Prediction & Probability Scoring
yhat = model.predict(x_test_sample)
probabilities = model.predict_proba(x_test_sample)

# 6. Print Results Safely (Removed the int() casting on the string classification output)
print("\n--- Execution Results ---")
print(f"Prediction Class Output: {yhat[0]}") 
print(f"Probability Profile: Class [{model.classes_[0]}] = {probabilities[0][0]:.4f}")
print(f"Probability Profile: Class [{model.classes_[1]}] = {probabilities[0][1]:.4f}")

