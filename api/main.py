
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title = "Loan Prediction API",
    version = "1.0"
)

# Load the trained model
model = joblib.load("./model/loan_default.pkl")

# --- DIAGNOSTIC STARTUP PRINT ---
print("\n" + "="*50)
print("TARGET MODEL DIAGNOSTICS")
print(f"Total features expected by tree: {len(model.feature_names_in_)}")
print(f"Model target classes identified: {list(model.classes_)}")
print("Exact feature names order:")
for idx, name in enumerate(model.feature_names_in_):
    print(f"  {idx + 1}. {name}")
print("="*50 + "\n")

class LoanInput(BaseModel):
    current_loan_amount: float = Field(alias="Current Loan Amount")
    term: float = Field(alias="Term")
    credit_score: float = Field(alias="Credit Score")
    annual_income: float = Field(alias="Annual Income")
    years_in_current_job: float = Field(alias="Years in current job")
    home_ownership: float = Field(alias="Home Ownership")
    purpose: float = Field(alias="Purpose")
    monthly_debt: float = Field(alias="Monthly Debt")
    years_of_credit_history: float = Field(alias="Years of Credit History")
    months_since_last_delinquent: float = Field(alias="Months since last delinquent")
    number_of_open_accounts: float = Field(alias="Number of Open Accounts")
    number_of_credit_problems: float = Field(alias="Number of Credit Problems")
    current_credit_balance: float = Field(alias="Current Credit Balance")
    maximum_open_credit: float = Field(alias="Maximum Open Credit")
    bankruptcies: float = Field(alias="Bankruptcies")
    tax_liens: float = Field(alias="Tax Liens")

@app.get("/")
def home():
    return {
        "message": "Loan Prediction API"
    }

@app.post("/predict")
def predict(data: LoanInput):
    # 1. Dump Pydantic payload utilizing structural aliases
    input_dict = data.model_dump(by_alias=True)
    
    # 2. Build Single-Row DataFrame
    input_data = pd.DataFrame([input_dict])
    
    # 3. Live Server Terminal Check
    print("\n[INCOMING REQUEST]")
    
    # 4. Reindex column positions to match training model layout exactly
    input_data = input_data[model.feature_names_in_]

    # 5. Execute Model Inference
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0] # Grab prediction confidence array row
    
    # Extract clean text string prediction ('Charged Off' or 'Fully Paid')
    pred_val = str(prediction[0])
    
    # Dynamically extract labels from classes metadata
    class_0_label = str(model.classes_[0]) # 'Charged Off'
    class_1_label = str(model.classes_[1]) # 'Fully Paid'
    
    print(f"Output: Prediction={pred_val} | Confidences: {class_0_label}={probabilities[0]:.4f}, {class_1_label}={probabilities[1]:.4f}")

    # 6. Return response matching exact dynamic class probabilities
    return {
        "prediction": pred_val,
        f"probability_{class_0_label.lower().replace(' ', '_')}": float(probabilities[0]),
        f"probability_{class_1_label.lower().replace(' ', '_')}": float(probabilities[1])
    }
