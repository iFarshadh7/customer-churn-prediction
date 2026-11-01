
import joblib
import pandas as pd
from pathlib import Path


# مسیر فایل مدل
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "churn_model_bundle.joblib"
)


def load_model():
    """Load the trained model and threshold."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["threshold"]


def predict_churn(customer_data):
    """
    Predict customer churn.

    Parameters
    ----------
    customer_data : pandas.DataFrame
        Customer information with the expected feature columns.

    Returns
    -------
    dict
        Churn prediction and churn probability.
    """
    model, threshold = load_model()

    probability = model.predict_proba(customer_data)[:, 1][0]

    prediction = int(probability >= threshold)

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability),
        "threshold": float(threshold)
    }
