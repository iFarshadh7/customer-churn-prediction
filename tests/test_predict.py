
import pandas as pd
from src.predict import predict_churn

def test_predict_churn_returns_valid_result():
    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }])

    result = predict_churn(sample)

    assert result["churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == 0.35
