# Customer Churn Prediction

## Project Overview

This project predicts customer churn using machine learning.
The goal is to identify customers who are likely to leave a
telecommunications company.

## Dataset

IBM Telco Customer Churn Dataset

- 7,043 customer records
- 21 columns
- Target variable: Churn

## Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Data Preprocessing
4. Model Training
5. Model Evaluation
6. Threshold Optimization
7. Model Interpretation

## Machine Learning Models

- Dummy Classifier
- Logistic Regression
- Random Forest

The selected model is Logistic Regression.

## Model Performance

Test-set results with threshold = 0.35:

| Metric | Score |
|---|---:|
| Accuracy | 0.7644 |
| Precision | 0.5432 |
| Recall | 0.7059 |
| F1-score | 0.6140 |
| ROC-AUC | 0.8421 |

## Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 813 | 222 |
| Actual Churn | 110 | 264 |

## Key Findings

- Customer tenure is an important predictive feature.
- Month-to-month contracts are associated with higher churn.
- Two-year contracts are associated with lower churn.
- Fiber optic customers show higher churn rates.

These findings represent associations, not causal relationships.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Repository Structure

- data/
- notebooks/
- models/
- src/
- tests/

## Future Improvements

- FastAPI prediction API
- Unit testing
- Docker
- GitHub Actions

## Evaluation Note

The test set was consulted during earlier model comparisons
and threshold exploration. Therefore, these metrics should be
considered indicative rather than a fully untouched final evaluation.
