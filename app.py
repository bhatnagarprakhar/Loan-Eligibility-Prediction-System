# from flask import Flask, request, render_template_string
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load all required artifacts
# model = joblib.load("loan_prediction_model.pkl")
# scaler = joblib.load("scaler.pkl")
# feature_names = joblib.load("feature_names.pkl")  # Saved during training

# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Loan Eligibility Predictor</title>
#     <style>
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             margin: 0;
#             padding: 20px;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             min-height: 100vh;
#         }
#         .container {
#             background: rgba(255, 255, 255, 0.95);
#             border-radius: 15px;
#             box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
#             padding: 30px;
#             width: 100%;
#             max-width: 500px;
#         }
#         h2 {
#             color: #333;
#             text-align: center;
#             margin-bottom: 25px;
#         }
#         .form-group {
#             margin-bottom: 20px;
#         }
#         label {
#             display: block;
#             margin-bottom: 8px;
#             font-weight: 600;
#             color: #555;
#         }
#         input, select {
#             width: 100%;
#             padding: 12px;
#             border: 2px solid #ddd;
#             border-radius: 8px;
#             font-size: 16px;
#             transition: border-color 0.3s;
#         }
#         input:focus, select:focus {
#             border-color: #667eea;
#             outline: none;
#         }
#         button {
#             background: linear-gradient(to right, #667eea, #764ba2);
#             color: white;
#             border: none;
#             padding: 14px;
#             width: 100%;
#             border-radius: 8px;
#             font-size: 16px;
#             font-weight: 600;
#             cursor: pointer;
#             transition: opacity 0.3s;
#         }
#         button:hover {
#             opacity: 0.9;
#         }
#         .result {
#             margin-top: 25px;
#             padding: 15px;
#             border-radius: 8px;
#             text-align: center;
#             font-weight: 600;
#             font-size: 18px;
#         }
#         .approved {
#             background-color: #d4edda;
#             color: #155724;
#         }
#         .rejected {
#             background-color: #f8d7da;
#             color: #721c24;
#         }
#         .error {
#             background-color: #fff3cd;
#             color: #856404;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h2>Loan Eligibility Predictor</h2>
#         <form action="/predict" method="post">
#             <div class="form-group">
#                 <label for="income">Applicant Income (₹)</label>
#                 <input type="number" id="income" name="income" min="0" required>
#             </div>
            
#             <div class="form-group">
#                 <label for="coapplicant_income">Co-applicant Income (₹)</label>
#                 <input type="number" id="coapplicant_income" name="coapplicant_income" min="0" required>
#             </div>
            
#             <div class="form-group">
#                 <label for="loan_amount">Loan Amount (₹)</label>
#                 <input type="number" id="loan_amount" name="loan_amount" min="0" required>
#             </div>
            
#             <div class="form-group">
#                 <label for="loan_term">Loan Term (days)</label>
#                 <input type="number" id="loan_term" name="loan_term" min="1" value="360" required>
#             </div>
            
#             <div class="form-group">
#                 <label for="credit_score">Credit Score (300-850)</label>
#                 <input type="number" id="credit_score" name="credit_score" min="300" max="850" required>
#             </div>
            
#             <div class="form-group">
#                 <label for="gender">Gender</label>
#                 <select id="gender" name="gender" required>
#                     <option value="Male">Male</option>
#                     <option value="Female">Female</option>
#                 </select>
#             </div>
            
#             <div class="form-group">
#                 <label for="married">Marital Status</label>
#                 <select id="married" name="married" required>
#                     <option value="Yes">Married</option>
#                     <option value="No">Single</option>
#                 </select>
#             </div>
            
#             <div class="form-group">
#                 <label for="self_employed">Employment Type</label>
#                 <select id="self_employed" name="self_employed" required>
#                     <option value="No">Salaried</option>
#                     <option value="Yes">Self-Employed</option>
#                 </select>
#             </div>
            
#             <button type="submit">Check Eligibility</button>
#         </form>
        
#         {% if result %}
#         <div class="result {{ result_class }}">
#             {{ result }}
#         </div>
#         {% endif %}
#     </div>
# </body>
# </html>
# """
# def predict_eligibility(data):
#     try:
#         # 1. Set Credit_History (adjust threshold if needed)
#         credit_history = 1 if data['credit_score'] >= 600 else 0
        
#         # 2. Ensure CoapplicantIncome isn't 0 (if training data had co-applicants)
#         coapplicant_income = data['coapplicant_income'] if data['coapplicant_income'] > 0 else 1000
        
#         # 3. Create input DataFrame with EXACTLY the same columns as training
#         input_data = pd.DataFrame([[
#             data['income'],
#             coapplicant_income,
#             credit_history,
#             data['loan_amount'],
#             data['loan_term'],
#             1 if data['gender'] == "Male" else 0,
#             1 if data['married'] == "Yes" else 0,
#             1 if data['self_employed'] == "Yes" else 0
#         ]], columns=feature_names)
        
#         # 4. Debug: Print input data before scaling
#         print("\n--- Input Data Before Scaling ---")
#         print(input_data)
        
#         # 5. Scale the data
#         scaled_data = scaler.transform(input_data)
        
#         # 6. Debug: Print scaled data
#         print("\n--- Scaled Input Data ---")
#         print(pd.DataFrame(scaled_data, columns=feature_names))
        
#         # 7. Predict
#         prediction = model.predict(scaled_data)[0]
        
#         return {
#             'result': "✅ Approved!" if prediction == 1 else "❌ Rejected.",
#             'result_class': 'approved' if prediction == 1 else 'rejected'
#         }
    
#     except Exception as e:
#         return {
#             'result': f"⚠️ Error: {str(e)}",
#             'result_class': 'error'
#         }
# @app.route('/', methods=['GET'])
# def home():
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Validate and convert form data
#         form_data = {
#             'income': float(request.form['income']),
#             'coapplicant_income': float(request.form['coapplicant_income']),
#             'loan_amount': float(request.form['loan_amount']),
#             'loan_term': float(request.form['loan_term']),
#             'credit_score': float(request.form['credit_score']),
#             'gender': request.form['gender'],
#             'married': request.form['married'],
#             'self_employed': request.form['self_employed']
#         }

#         # Validate numerical inputs
#         for field in ['income', 'coapplicant_income', 'loan_amount', 'loan_term']:
#             if form_data[field] < 0:
#                 raise ValueError(f"{field} cannot be negative")
        
#         if not (300 <= form_data['credit_score'] <= 850):
#             raise ValueError("Credit score must be between 300 and 850")

#         # Get prediction
#         prediction = predict_eligibility(form_data)
#         return render_template_string(HTML_TEMPLATE, 
#                                   result=prediction['result'],
#                                   result_class=prediction['result_class'])

#     except ValueError as ve:
#         return render_template_string(HTML_TEMPLATE, 
#                                    result=f"Invalid input: {str(ve)}",
#                                    result_class='error')
#     except KeyError as ke:
#         return render_template_string(HTML_TEMPLATE, 
#                                    result=f"Missing field: {str(ke)}",
#                                    result_class='error')
#     except Exception as e:
#         return render_template_string(HTML_TEMPLATE, 
#                                    result=f"An error occurred: {str(e)}",
#                                    result_class='error')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained pipeline and feature names
model = joblib.load('loan_model_pipeline.pkl')
feature_names = joblib.load('feature_names.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Eligibility Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            padding: 30px;
            width: 100%;
            max-width: 600px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #34495e;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            transition: border 0.3s;
        }
        input:focus, select:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            background: linear-gradient(to right, #3498db, #2c3e50);
            color: white;
            border: none;
            padding: 14px;
            width: 100%;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        button:hover {
            opacity: 0.9;
        }
        .result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .approved {
            background-color: #d4edda;
            color: #155724;
        }
        .rejected {
            background-color: #f8d7da;
            color: #721c24;
        }
        .error {
            background-color: #fff3cd;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Loan Eligibility Predictor</h1>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="applicant_income">Applicant Income (₹)</label>
                <input type="number" id="applicant_income" name="applicant_income" min="0" required>
            </div>
            
            <div class="form-group">
                <label for="coapplicant_income">Co-applicant Income (₹)</label>
                <input type="number" id="coapplicant_income" name="coapplicant_income" min="0" required>
            </div>
            
            <div class="form-group">
                <label for="loan_amount">Loan Amount (₹)</label>
                <input type="number" id="loan_amount" name="loan_amount" min="0" required>
            </div>
            
            <div class="form-group">
                <label for="loan_term">Loan Term (months)</label>
                <input type="number" id="loan_term" name="loan_term" min="1" value="360" required>
            </div>
            
            <div class="form-group">
                <label for="credit_history">Credit History</label>
                <select id="credit_history" name="credit_history" required>
                    <option value="1">Good (≥ 650)</option>
                    <option value="0">Poor (< 650)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="married">Marital Status</label>
                <select id="married" name="married" required>
                    <option value="1">Married</option>
                    <option value="0">Single</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="self_employed">Employment Type</label>
                <select id="self_employed" name="self_employed" required>
                    <option value="0">Salaried</option>
                    <option value="1">Self-Employed</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="property_area">Property Area</label>
                <select id="property_area" name="property_area" required>
                    <option value="0">Rural</option>
                    <option value="1">Semiurban</option>
                    <option value="1">Urban</option>
                </select>
            </div>
            
            <button type="submit">Check Eligibility</button>
        </form>
        
        {% if result %}
        <div class="result {{ result_class }}">
            {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

def prepare_input(data):
    """Convert form data to model input format"""
    # Calculate derived features
    total_income = float(data['applicant_income']) + float(data['coapplicant_income'])
    loan_amount = float(data['loan_amount'])
    emi_ratio = loan_amount / total_income if total_income > 0 else 0
    
    # Property area encoding
    property_area = data['property_area']
    semiurban = 1 if property_area == "1" else 0
    urban = 1 if property_area == "1" else 0
    
    # Create input dictionary
    input_data = {
        'ApplicantIncome': float(data['applicant_income']),
        'CoapplicantIncome': float(data['coapplicant_income']),
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(data['loan_term']),
        'Credit_History': int(data['credit_history']),
        'Gender': int(data['gender']),
        'Married': int(data['married']),
        'Self_Employed': int(data['self_employed']),
        'TotalIncome': total_income,
        'EMI_Ratio': emi_ratio,
        'Property_Area_Semiurban': semiurban,
        'Property_Area_Urban': urban
    }
    
    # Convert to DataFrame with correct column order
    return pd.DataFrame([input_data], columns=feature_names)

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Prepare input data
        input_df = prepare_input(request.form)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Prepare result
        result = "Congratulations! Your loan is approved." if prediction == 1 \
                else "Sorry, your loan application was not approved."
        result_class = "approved" if prediction == 1 else "rejected"
        
        return render_template_string(HTML_TEMPLATE, result=result, result_class=result_class)
    
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        return render_template_string(HTML_TEMPLATE, result=error_msg, result_class="error")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)