

# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
 
# 2. Read Data
credit_df = pd.read_csv("./data/credit_train.csv", header = 0, sep = ',')
 
# 3. Data Processing & Cleansing
credit_df['Months since last delinquent'] = credit_df['Months since last delinquent'].fillna(0)
credit_df['Credit Score'] = credit_df[credit_df['Credit Score'] >= 900]['Credit Score'] / 10
credit_df['Credit Score'] = credit_df['Credit Score'].fillna(round(credit_df['Credit Score'].mean()))
credit_df['Annual Income'] = credit_df['Annual Income'].fillna(round(credit_df['Annual Income'].mean()))
 
# Clean 'Years in current job' safely
credit_df['Years in current job'] = credit_df['Years in current job'].astype(str).str.replace(' years', '').str.replace(' year', '').str.replace('< 1', '0.5').str.replace('+', '')
credit_df['Years in current job'] = pd.to_numeric(credit_df['Years in current job'], errors='coerce')
credit_df['Years in current job'] = credit_df['Years in current job'].fillna(credit_df['Years in current job'].median())
credit_df.dropna(inplace = True)
 
# 4. Drop Columns
credit_df.drop(['Loan ID', 'Customer ID'], axis = 1, inplace=True)
 
# 5. Data Pre-Processing - FIXED: Select columns by structural dtype to catch all string arrays
binary_cols = credit_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Ensure 'Loan Status' (target variable) isn't grouped with feature matrix variables
if 'Loan Status' in binary_cols:
    binary_cols.remove('Loan Status')

print("Successfully detected and encoding categorical columns:", binary_cols)
 
# Encode categories deterministically to numeric code representations
for col in binary_cols:
    credit_df[col] = credit_df[col].astype('category').cat.codes.astype(float)

# Encode Target variable ('Loan Status') if it contains text metrics like "Fully Paid"
if credit_df['Loan Status'].dtype in ['object', 'category']:
    credit_df['Loan Status'] = credit_df['Loan Status'].astype('category').cat.codes
 
# 6. Features & Target
X = credit_df.drop('Loan Status', axis = 1)
Y = credit_df['Loan Status']

# Convert entire feature matrix to floats to eliminate SMOTE typing discrepancies
X = X.astype(float)
 
# 7. Add SMOTE - This will now execute seamlessly without crashing
print("Applying SMOTE balancing transformation across features...")
smote = SMOTE(random_state=2)
transformed_feature, transformed_label = smote.fit_resample(X, Y)
 
# 9. Split Data
X_train, X_test, Y_train, Y_test = train_test_split(transformed_feature, transformed_label, test_size = 0.2, random_state = 2)
 
# 10. Decision Tree Classifier
print("Training the Decision Tree Classifier pipeline...")
clf_tree_best = DecisionTreeClassifier(ccp_alpha = 0.001, criterion = 'gini', max_depth = 20, random_state = 2)
clf_tree_best.fit(X_train, Y_train)
 
# Save model artifact out
import os
os.makedirs("./model", exist_ok=True)
joblib.dump(clf_tree_best, "./model/loan_default.pkl")
print("\nModel Saved Successfully with proper numeric Encodings!")

# Save actual encoded mathematical samples to verify deployment pipelines
os.makedirs("./data", exist_ok=True)
X_test.head(10).to_csv("./data/x_test_sample.csv", index=False)
print("Saved testing vectors output to './data/x_test_sample.csv'")
