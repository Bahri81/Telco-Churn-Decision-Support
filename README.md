# Telco Churn Prediction: End-to-End Enterprise Project Report

## Project Objective
The primary goal of this project is to systematically analyze and predict customer churn within the telecommunications sector. Beyond merely identifying which profiles are likely to leave, this framework delivers a concrete, data-driven blueprint for business stakeholders to optimize retention strategies and maximize effective budget utilization.

## Data Challenges & Methodology
The underlying dataset presents classic tabular challenges, featuring a severe class imbalance alongside a dense combination of demographic, financial, and raw geographic attributes.

### Avoiding Data Leakage
To maintain strict methodological honesty and prevent data leakage, all preprocessing pipelines—including hyperparameter tuning and SMOTE (Synthetic Minority Over-Sampling Technique) for class imbalance management—were strictly locked within the training cross-validation loops.

### Experimentation Workflow
* **Baseline Classifier Evaluation:** We initially compared the performance of four distinct classifiers—AdaBoost, Linear Discriminant Analysis (LDA), Gradient Boosting Classifier (GBC), and Multi-Layer Perceptron (MLP)—using 5-Fold Stratified Cross-Validation on the training dataset.
* **Hyperparameter Optimization:** Subsequently, we utilized PyCaret’s optimization frameworks to fine-tune the hyperparameters of our candidate models, specifically targeting the maximization of the F1-Score.
* **Advanced Ensembling (Stacking):** We constructed a hierarchical Stacking pipeline using GBC, AdaBoost, and LDA as our foundation estimators, with a Logistic Regression model acting as the "referee" meta-model.

---

## Key Performance Metrics & Model Selection

### A Lesson in Occam’s Razor
While the complex Stacking model appeared superior during cross-validation (achieving a 65% F1-Score), it ultimately overfitted by memorizing the training patterns, dropping to 61.75% on the holdout test set. Consequently, the **Tuned Gradient Boosting Classifier (GBC)** was selected as our final production model due to its superior robustness and generalization capabilities, reaching a **0.6243 F1-Score** on the completely untouched holdout dataset.

### Probability Threshold Optimization
Rather than blindly accepting the default 0.50 decision boundary, we optimized the probability threshold to locate the exact sweet spot where Precision and Recall balance out. Shifting the decision threshold to **0.56** proved highly effective; it kept our recall intact while reducing false alarms, successfully increasing Precision from 52.5% to **54.9%**.

### Operational Trade-Offs
Depending on strategic business priorities, an alternative model choice could be justified: if the business objective is to catch *every single* churning customer regardless of the financial cost of false alarms, **Tuned LDA** exhibited the highest recall performance. However, for a balanced, cost-effective enterprise deployment, **Tuned GBC** remains the optimal choice.

---

## Model Explainability & Business Insights (SHAP)
Using SHAP (SHapley Additive exPlanations), we extracted actionable strategic insights from the model's internal decision-making process:
* **Structural Lock-in:** Contract type (0.6822) and Tenure Months (0.6793) emerged as the absolute most critical drivers of customer retention.
* **The Price Paradox:** Surprisingly, Monthly Charges (0.1050) sits relatively low in feature importance. The model reveals that customers do not leave purely because of price; they churn primarily when they lack a structural or contractual commitment to the provider.
* **The Fiber-Optic Red Flag:** Internet Service via Fiber Optic (0.3298) significantly *increases* churn risk. This serves as a massive operational red flag, pointing to potential price shocks after promotional periods end or localized infrastructure/service quality issues.
* **Geographic Sanity Check:** Geographical coordinates (Latitude/Longitude) generated zero predictive impact, proving uniform service quality and consistency across all analyzed regions.

---

## MLOps Pipeline & Deployment Stages

### 1. Experiment Tracking (MLflow)
All model experiments, hyperparameter configurations, and SHAP analyses were systematically tracked and managed using **MLflow Tracking**. Once the repository is downloaded, the complete interactive experiment history can be reviewed locally through the MLflow UI by executing the `mlflow ui` command.

### 2. Live Visual Production (Streamlit)
To deliver these predictive capabilities to non-technical business stakeholders, the finalized pipeline has been deployed as a live interactive web application using **Streamlit**. 
* **Interactive Single Profile Simulator:** Allows real-time risk analysis for individual customer profiles.
* **High-Volume Bulk Prediction Engine:** Enables rapid batch portfolio forecasting via Excel/CSV uploads.

> ### 🌐 Live Demonstration
> [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-churn-decision-support-axexvfsw7za7qckkax8rds.streamlit.app)

### 3. Production-Ready Architecture (FastAPI & Docker)
To complement the visual frontend with an enterprise-grade backend, a robust microservice infrastructure has been integrated into the repository:
* **Microservice API (`main.py` via FastAPI):** The selected model is exposed as a high-performance REST API using FastAPI and Uvicorn. This simulates how internal enterprise platforms (like CRMs or billing software) automatically request real-time Churn predictions via JSON payloads. The API dynamically mirrors the exact mathematical transformations (Z-score scaling and categorical dummy mapping) applied during training.
* **Containerization (`Dockerfile`):** The FastAPI service, Python environment, and PyCaret prediction pipeline are packaged into an isolated Docker container, ensuring consistent, error-free execution across cloud platforms like AWS, Google Cloud, and Azure.

---

## How to Run the Production Environment (Local Docker)

If you wish to deploy and test the enterprise API locally using Docker, execute the following commands in your terminal:

```bash
# 1. Build the Docker Image
docker build -t telco-churn-api .

# 2. Run the Containerized API
docker run -d -p 80:80 telco-churn-api
