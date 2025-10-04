import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Or any other model you're using
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("C:\\Users\\ASUS\\Desktop\\pythonproject\\LoanApprovalPrediction.csv")

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=[object]).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Handle missing values (impute with median for numerical and mode for categorical)
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# One-hot encoding for categorical variables (if required, or use LabelEncoder)
# data = pd.get_dummies(data, drop_first=True)  # Optional, if using one-hot encoding

# Split data into features (X) and target (Y)
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Scaling numerical data using StandardScaler
scaler = StandardScaler()

# Fit scaler on the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data with the same scaler
X_test_scaled = scaler.transform(X_test)

# Now fit a model (e.g., RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train_scaled, Y_train)

# After training, make predictions
predictions = model.predict(X_test_scaled)

# Evaluate model (for example, accuracy)
from sklearn.metrics import accuracy_score
print("Model Accuracy: ", accuracy_score(Y_test, predictions))

# If you need to predict for new data, apply the same transformations:
def predict_loan(income, loan_amount, credit_score, employment_status, gender, married):
    # Create a DataFrame for the new data point
    input_data = pd.DataFrame({
        'Income': [income],
        'LoanAmount': [loan_amount],
        'CreditScore': [credit_score],
        'EmploymentStatus': [employment_status],
        'Gender': [gender],
        'Married': [married]
    })
    
    # Apply the same scaling
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the trained model
    result = model.predict(input_data_scaled)
    return result[0]  # Return the predicted loan status

# Example prediction
result = predict_loan(50000, 200000, 700, 1, 0, 1)
print(f"Loan Approval Prediction: {result}")
