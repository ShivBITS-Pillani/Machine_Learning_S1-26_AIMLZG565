# Online Shopping Purchase Intent - Machine Learning Assignment 2

This repository supports a reproducible comparison of five classifiers.
All final metrics and observations must be validated on BITS Virtual Lab.

## a. Problem statement

Predict whether a customer browsing an e-commerce website will finish the
session with a purchase, using 17 recorded session features (Revenue target).
The five classifiers are fitted to a shared stratified training split and are
compared on an untouched held-out test split.

## b. Dataset description

- Source: UCI Machine Learning Repository, dataset ID 468.
- Dataset name: Online Shoppers Purchasing Intention Dataset.
- Source URL: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Instances: 12,330 shopping sessions.
- Source predictor features: 17, with a binary target named Revenue.
- No-purchase class: 10,422 sessions; purchase class: 1,908 sessions.
- Distribution is imbalanced; Accuracy alone can be misleading.
- License: CC BY 4.0; credit the dataset's original contributors.

## c. GitHub Repository Link

PENDING: replace with the publicly accessible GitHub repository URL.

## d. Models used - Comparison table

Run the training script and fill this table with YOUR actual metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | - | - | - | - | - | - |
| Decision Tree | - | - | - | - | - | - |
| KNN | - | - | - | - | - | - |
| Naive Bayes | - | - | - | - | - | - |
| Support Vector Machine | - | - | - | - | - | - |

## Model performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | PENDING: your observation |
| Decision Tree | PENDING: your observation |
| KNN | PENDING: your observation |
| Naive Bayes | PENDING: your observation |
| Support Vector Machine | PENDING: your observation |
| Overall winner for your dataset? | PENDING: your evidence-based choice |

## Training and deployment methodology

- Train/test split: 80%/20%, stratified by Revenue, random_state=42.
- Preprocessing: median imputation and scaling for numeric predictors;
  categorical imputation and one-hot encoding for categorical predictors.
- All preprocessing is fitted on training data within each saved Pipeline.
- Five classifiers: Logistic Regression, Decision Tree, KNN, Gaussian Naive
  Bayes and RBF Support Vector Machine.
- Evaluation: Accuracy, ROC AUC, Precision, Recall, F1 and MCC.
- Live app loads the trained Pipelines without fitting on uploaded test data.

## Reproduce in BITS Virtual Lab

1. Install packages: python -m pip install -r requirements.txt
2. Get UCI data: python download_dataset.py (or manual download).
3. Train and save test set: python train_models.py
4. Test app: streamlit run app.py
5. Upload test_data.csv and compare all models.
6. Review and write your own model observations.
7. Update submission_details.json and run python prepare_submission.py.

## Files

- train_models.py: model fitting and saved hold-out test set.
- common.py: shared schema and evaluation metrics.
- app.py: interactive Streamlit evaluation, no retraining.
- model/*.joblib: five fitted preprocessing + model Pipelines (after training).
- test_data.csv: held-out labelled test data (after training).
- model_comparison.csv: six-metric results table (after training).
- requirements.txt: application dependencies.
- prepare_submission.py: build final README and one-PDF submission.

## Attribution

Sakar, C. and Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset.
UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.
