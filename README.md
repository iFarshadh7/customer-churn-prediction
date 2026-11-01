
# Customer Churn Prediction

![ML Project Tests](https://github.com/iFarshadh7/customer-churn-prediction/actions/workflows/tests.yml/badge.svg)

**An End-to-End Machine Learning Project for Predicting Customer Churn**

This project develops a machine learning solution to predict customer churn in the telecommunications industry. It covers data analysis, preprocessing, model development, evaluation, model interpretation, API development, and automated testing.

---

## 1. Project Overview

Customer churn occurs when customers stop using a company's services.

The objective of this project is to identify customers who are likely to leave a telecommunications company using historical customer data.

**Business Goal:** Help businesses identify customers at risk of churn and support data-driven customer retention strategies.

## 2. Dataset

The project uses the **IBM Telco Customer Churn Dataset**.

| Description | Value |
|---|---:|
| Total records | 7,043 |
| Input features | 19 |
| Target variable | `Churn` |
| Problem type | Binary Classification |

The dataset includes customer demographics, account information, service subscriptions, contract types, and payment methods.

Target encoding:

- `No` → `0`
- `Yes` → `1`

## 3. Project Workflow

The project follows an end-to-end machine learning workflow:

1. **Exploratory Data Analysis (EDA)**
   - Dataset inspection
   - Missing value analysis
   - Churn distribution analysis
   - Customer segment analysis

2. **Data Preprocessing**
   - Data type correction
   - Missing value handling
   - Numerical feature scaling
   - Categorical feature encoding

3. **Model Development**
   - Logistic Regression
   - Scikit-learn Pipeline
   - ColumnTransformer
   - Stratified data splitting

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix

5. **Model Interpretation**
   - Logistic Regression coefficients
   - Identification of important churn-related patterns

6. **Deployment and Testing**
   - Model serialization with Joblib
   - Prediction module
   - FastAPI REST API
   - Pytest automated tests
   - GitHub Actions CI

## 4. Model Performance

The final Logistic Regression model uses a classification threshold of **0.35**, selected based on validation-set F1-score.

| Metric | Score |
|---|---:|
| Accuracy | 76.44% |
| Precision | 54.32% |
| Recall | 70.59% |
| F1-score | 61.40% |
| ROC-AUC | 84.21% |

### Threshold Selection

The default classification threshold is 0.50. A threshold of 0.35 was selected to improve the balance between precision and recall based on validation results.

This threshold increases the number of correctly identified churn customers, while also producing more false positives.

**Evaluation limitation:** The test set was consulted during earlier model experiments. Therefore, the reported test metrics are exploratory and should not be considered a final evaluation on a completely untouched holdout set.

## 5. Key Findings

Exploratory analysis and model interpretation revealed several patterns:

- Customers with month-to-month contracts had substantially higher churn rates.
- Customers without online security and technical support services showed higher churn rates.
- Customers using electronic check payments had higher observed churn rates.
- Customer tenure was an important predictive feature.

These findings describe associations in the dataset and do not establish causation.

## 6. REST API

The trained model is integrated into a REST API using **FastAPI**.

### Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API from the project root:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Prediction Response

The `/predict` endpoint returns:

- Churn prediction (`Yes` or `No`)
- Churn probability
- Classification threshold

## 7. Automated Testing and CI

The project uses **Pytest** for automated testing.

GitHub Actions automatically runs the test suite when code is pushed to the `main` branch or when a pull request is created.

![GitHub Actions](https://github.com/iFarshadh7/customer-churn-prediction/actions/workflows/tests.yml/badge.svg)

## 8. Project Structure

```text
customer-churn-prediction/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── data/
│
├── models/
│   └── churn_model_bundle.joblib
│
├── notebooks/
│   └── 01_customer_churn_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── predict.py
│
├── tests/
│   └── test_predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 9. Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model | Logistic Regression |
| Model Serialization | Joblib |
| API | FastAPI |
| Testing | Pytest |
| Version Control | Git, GitHub |
| Continuous Integration | GitHub Actions |

## 10. Future Improvements

- Evaluate additional classification algorithms.
- Perform hyperparameter tuning.
- Apply cross-validation.
- Evaluate the model on a completely untouched test set.
- Explore SHAP for model explainability.
- Deploy the API to a cloud platform.
- Build an interactive churn monitoring dashboard.

---

## Author

**Farshad**

GitHub: [@iFarshadh7](https://github.com/iFarshadh7)

---

*Developed as a machine learning portfolio project demonstrating practical skills in data analysis, predictive modeling, model evaluation, API development, and automated testing.*
