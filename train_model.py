# 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('train.csv')

# Data Cleaning
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace('3+', '36').astype(float)

# Handle missing values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Feature Engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI_Ratio'] = df['LoanAmount'] / df['TotalIncome'].replace(0, 1)  # Avoid division by zero

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

# Define features and target
features = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Gender', 'Married', 'Self_Employed', 'TotalIncome', 'EMI_Ratio',
    'Property_Area_Semiurban', 'Property_Area_Urban'
]
X = df[features]
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipeline
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 'EMI_Ratio']
binary_features = ['Credit_History', 'Gender', 'Married', 'Self_Employed', 'Property_Area_Semiurban', 'Property_Area_Urban']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('passthrough', 'passthrough', binary_features)
    ])

# Create and train model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Evaluate model
print("Training Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Save artifacts
joblib.dump(model, 'loan_model_pipeline.pkl')
joblib.dump(features, 'feature_names.pkl')

print("\nModel training complete! Saved:")
print("- loan_model_pipeline.pkl (full trained pipeline)")
print("- feature_names.pkl (feature order reference)")