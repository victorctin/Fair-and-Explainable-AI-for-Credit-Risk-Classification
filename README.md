# Fair and Explainable AI for Credit Risk Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)
![AIF360](https://img.shields.io/badge/Fairness-AIF360-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A multiclass machine learning system that classifies credit applicants by risk level, with built-in fairness analysis and explainable AI.

**Live Demo:** [Open on Streamlit Cloud](https://fair-and-explainable-ai-for-credit-risk-classification-iwyjlgb.streamlit.app/)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Target Classes](#target-classes)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Notebooks](#notebooks)
- [Fairness Analysis](#fairness-analysis)
- [Explainability with SHAP](#explainability-with-shap)
- [Author](#author)

---

## Project Overview

This project builds a multiclass credit risk classification model using supervised machine learning. It goes beyond standard binary loan approval systems by segmenting applicants into three risk categories. The model is trained on historical financial and behavioural data.

A core design principle of the project is responsibility. The system is evaluated not only for predictive accuracy but also for fairness across sensitive demographic attributes and interpretability through explainable AI techniques.

The project includes a Streamlit web application where users can input applicant data and receive a real-time risk classification with SHAP-based explanations.

---

## Problem Statement

Traditional credit scoring systems often rely on rigid scorecards or opaque models. These systems can produce biased outcomes, particularly for applicants from underrepresented groups. They also give no explanation for their decisions, which reduces trust and hinders regulatory compliance.

This project addresses three key challenges in credit risk modelling:

1. **Accuracy** — moving from binary to multiclass classification for more granular risk assessment
2. **Fairness** — measuring and reducing bias against sensitive attributes using IBM AIF360
3. **Transparency** — explaining individual predictions using SHAP values

---

## Target Classes

The model classifies each applicant into one of three risk levels:

| Class | Description |
|---|---|
| **Good Client** | Low probability of default. Strong repayment history and financial indicators. |
| **Low Risk** | Moderate credit profile. Some financial instability but manageable risk. |
| **High Risk** | Elevated probability of default. Requires stricter credit conditions or rejection. |

This segmentation supports more personalised lending decisions. Financial institutions can use it to set appropriate credit limits, adjust interest rates by risk tier, and identify high-risk profiles early.

---

## Key Features

- Multiclass classification (3 risk levels) instead of binary approval/rejection
- Class imbalance handling with SMOTE (Synthetic Minority Over-sampling Technique)
- Multiple models trained and compared: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Fairness evaluation using IBM AIF360 (disparate impact, equal opportunity difference)
- Feature importance and prediction explanation using SHAP
- Interactive prediction interface deployed on Streamlit Cloud
- SQL-based data exploration notebook included

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP, streamlit-shap |
| Fairness | IBM AIF360 |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Model Persistence | joblib |
| Web Application | Streamlit |

---

## Project Structure

```
Fair-and-Explainable-AI-for-Credit-Risk-Classification/
│
├── datasets/                          # Raw input datasets
├── final_datasets/                    # Processed and cleaned datasets
├── models/                            # Saved trained model files (.pkl / .joblib)
├── plots/                             # Generated visualisations and charts
├── utils/                             # Helper functions and preprocessing utilities
│
├── Credit_Risk_Classification_Model.ipynb   # Main modelling notebook
├── fairness_aif360.ipynb                    # Fairness analysis with IBM AIF360
├── SQL.ipynb                                # SQL-based exploratory data analysis
├── app.py                                   # Streamlit web application
├── fairness_metrics_summary.csv             # Summary of fairness evaluation results
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## How to Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/victorctin/Fair-and-Explainable-AI-for-Credit-Risk-Classification.git
cd Fair-and-Explainable-AI-for-Credit-Risk-Classification
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit app**

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Notebooks

### `Credit_Risk_Classification_Model.ipynb`

The main modelling notebook covers the full machine learning pipeline:

- Exploratory data analysis (EDA)
- Feature engineering and preprocessing
- SMOTE for class imbalance correction
- Model training: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix
- SHAP integration for global and local feature importance

### `fairness_aif360.ipynb`

Dedicated fairness analysis using IBM AIF360:

- Definition of privileged and unprivileged groups
- Measurement of fairness metrics: disparate impact, statistical parity difference, equal opportunity difference
- Results exported to `fairness_metrics_summary.csv`

### `SQL.ipynb`

SQL-based exploratory analysis of the credit dataset. Covers aggregations, filtering, and data profiling using Python's SQLite interface.

---

## Fairness Analysis

The project uses IBM AIF360 to evaluate model fairness across sensitive attributes. Key metrics include:

- **Disparate Impact** — ratio of positive outcomes between unprivileged and privileged groups (ideal value: 1.0)
- **Statistical Parity Difference** — difference in positive prediction rates between groups (ideal value: 0.0)
- **Equal Opportunity Difference** — difference in true positive rates between groups (ideal value: 0.0)

Results are summarised in `fairness_metrics_summary.csv`.

---

## Explainability with SHAP

The model uses SHAP (SHapley Additive exPlanations) to explain predictions at both global and individual level.

- **Global explanations** — identify which features drive risk classification across the full dataset
- **Local explanations** — show why a specific applicant received a particular risk label
- SHAP waterfall and summary plots are included in the Streamlit application

This makes the system suitable for use in regulated environments where model decisions must be auditable and explainable.

---

## Author

**Victor Pavel**



