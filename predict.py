import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('CVD_cleaned_v2.csv')

# Columns of interest
features = ['Sex', 'Age_Category', 'Height', 'Weight', 'BMI', 'Diabetes', 'Arthritis']
target = 'Heart_Disease'

X = df[features].copy()
y = df[target]

# Encode categorical variables
label_encoders = {}
yes_no_map = {1: 'Yes', 0: 'No'}
for col in ['Sex', 'Age_Category', 'Diabetes', 'Arthritis']:
    if col in ['Diabetes', 'Arthritis']:
        X[col] = X[col].map(yes_no_map)
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def predict_heart_disease(user_input_dict):
    """
    Predicts heart disease from user input.

    Expected input keys: 'Sex', 'Age_Category', 'Height', 'Weight', 'BMI', 'Diabetes', 'Arthritis'
    """
    input_df = pd.DataFrame([user_input_dict])

    # Encode categorical fields using the same encoders
    for col in ['Sex', 'Age_Category', 'Diabetes', 'Arthritis']:
        val = str(input_df[col].iloc[0])  # Force string
        try:
            encoded_val = label_encoders[col].transform([val])[0]
        except ValueError:
            raise ValueError(f"Unrecognized value '{val}' for column '{col}'. Expected one of: {label_encoders[col].classes_}")
        input_df.at[0, col] = encoded_val
    input_df = input_df[X.columns]  # Ensure column order

    pred_proba = rf_model.predict_proba(input_df)[0][1]
    pred_class = rf_model.predict(input_df)[0]

    return {
        "prediction": int(pred_class),
        "probability": round(pred_proba, 4)
    }

# AGE_MAPPING = {
#     "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4,
#     "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
#     "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
# }

# EXPECTED_FEATURES = [
#     'Sex', 'Age_Category', 'Height', 'Weight', 'BMI', 'Diabetes', 'Arthritis'
# ]

# def preprocess_input(input_dict):
#     df = pd.DataFrame([input_dict])

#     # Map all categorical fields
#     df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
#     df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
#     df['Arthritis'] = df['Arthritis'].map({'Yes': 1, 'No': 0})
#     df['Age_Category'] = df['Age_Category'].map(AGE_MAPPING)

#     # Check for any mapping errors
#     if df.isnull().any(axis=None):
#         print("⚠️ Warning: Missing or unmapped values in input:", df)

#     # Enforce correct order
#     df = df[EXPECTED_FEATURES]
#     print("Final input types:\n", df.dtypes)
#     return df

# def predict_cvd_risk(input_dict):
#     try:
#         print("🧪 Raw input:", input_dict)
#         X = preprocess_input(input_dict)
#         print("✅ Preprocessed input:\n", X)
#         prediction = model.predict(X)
#         result = prediction[0]  # Return as 0 or 1
#         if isinstance(result, str):
#             result = 1 if result.lower() == "yes" else 0
#         return int(result)
#     except Exception as e:
#         print("Prediction error:", e)
#         return None

