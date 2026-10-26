# Online Shopping Purchase Intent - Machine Learning Assignment 2

**Programme:** BITS Pilani WILP - M.Tech (Artificial Intelligence and Machine Learning)  
**Course:** Machine Learning  
**Student:** Shiv Nandan Yadav  
**Student ID:** 2026ak05694

This project compares five classification algorithms on one dataset and provides a Streamlit application for evaluating saved models using labelled test data.

- **GitHub repository:** https://github.com/ShivBITS-Pillani/Machine_Learning_S1-26_AIMLZG565/
- **Streamlit application:** https://machinelearnings1-26aimlzg565-ttyjozfktunv9zuxl59lv3.streamlit.app/

## a. Problem statement

The objective is to predict whether a visitor to an e-commerce website will complete a purchase during a browsing session. This is a **binary classification** problem in which the target variable, `Revenue`, represents either a purchase (`True` / `1`) or no purchase (`False` / `0`).

Five classifiers are trained on the same dataset and compared on the same held-out test set. The objective is to evaluate their performance using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). Because relatively few sessions end in purchases, model selection considers more than Accuracy alone.

## b. Dataset description

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository, dataset ID 468
- **Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Total instances:** 12,330 shopping sessions
- **Input features:** 17 (10 numerical and 7 categorical in this implementation)
- **Target:** `Revenue` (binary)
- **Class distribution:** 10,422 no-purchase sessions and 1,908 purchase sessions (approximately 84.53% and 15.47%)
- **Problem type:** Binary classification with an imbalanced target distribution

**Numerical features (10):** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, and `SpecialDay`.

**Categorical features (7):** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, and `Weekend`.

The dataset exceeds the assignment requirement of at least 500 instances and 12 features. The modelling code performs an 80% training / 20% testing split using `stratify=y` and `random_state=42`; the test set has **2,466 rows**.

## c. GitHub Repository Link

https://github.com/ShivBITS-Pillani/Machine_Learning_S1-26_AIMLZG565/

**Live Streamlit application:** https://machinelearnings1-26aimlzg565-ttyjozfktunv9zuxl59lv3.streamlit.app/

## d. Models used - Comparison table

All five models use the same training data and are evaluated on the same **held-out, labelled test set**. The following metrics are taken from the project's `model_comparison.csv` and are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8410 | 0.8932 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8463 | **0.9113** | 0.5024 | 0.8220 | 0.6236 | 0.5596 |
| KNN | **0.8723** | 0.8285 | **0.7055** | 0.3010 | 0.4220 | 0.4049 |
| Naive Bayes | 0.2887 | 0.7571 | 0.1752 | **0.9686** | 0.2967 | 0.1369 |
| Support Vector Machine | 0.8654 | 0.8979 | 0.5477 | 0.7513 | **0.6336** | **0.5639** |

**Metric interpretation:** Accuracy measures the fraction of correct predictions. AUC measures ranking discrimination across decision thresholds. Precision measures the proportion of predicted purchases that are correct, while Recall measures the proportion of actual purchases identified. F1 balances Precision and Recall; MCC accounts for all four outcomes of the confusion matrix and is informative for imbalanced data.

## Model performance observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy is 84.10%, AUC is 0.8932, Precision is 0.4913 and Recall is 0.7435. It identifies nearly three quarters of actual purchasers, although its lower Precision indicates that some sessions predicted as purchases do not convert. Its F1 of 0.5917 provides a useful linear baseline. The implementation uses class balancing to give the minority class greater importance. |
| **Decision Tree** | Decision Tree has the highest AUC (0.9113) and 82.20% Recall, with F1 of 0.6236 and MCC of 0.5596. It identifies more actual purchasers than SVM, at the cost of lower Precision (0.5024). A maximum depth and minimum leaf size limit complexity; however, training scores would be needed to make a direct claim about overfitting. |
| **KNN** | KNN achieves the highest Accuracy (87.23%) and Precision (0.7055), but its Recall is only 0.3010. Consequently, its F1 is 0.4220 and MCC is 0.4049. It is relatively conservative about predicting purchases and misses many real purchasers. High Accuracy alone is therefore not enough for this dataset. |
| **Naive Bayes** | Gaussian Naive Bayes achieves the highest Recall (0.9686), but its Precision is only 0.1752 and Accuracy is 28.87%. Its low F1 (0.2967) and MCC (0.1369) indicate many false purchase predictions. The conditional independence assumption may not fit correlated browsing features well, although the scores alone do not establish the exact cause. |
| **Support Vector Machine** | SVM obtains 86.54% Accuracy, 0.8979 AUC, 0.5477 Precision and 0.7513 Recall. It produces the highest F1 (0.6336) and MCC (0.5639) among the five classifiers. Its RBF kernel supports a nonlinear decision boundary and provides the strongest overall balance on these selected metrics. |
| **Overall winner for your dataset?** | **Support Vector Machine (SVM)** is selected because it achieves the best F1 (0.6336) and MCC (0.5639), with 86.54% Accuracy and 0.8979 AUC. These are helpful criteria for this imbalanced dataset. **Decision Tree is a close alternative** with superior AUC (0.9113) and Recall (0.8220), and could be preferable when detecting more actual purchasers is the highest priority. |

## Training and deployment methodology

1. **Data validation:** `common.py` checks that the CSV contains the 17 expected input features and the `Revenue` target, and converts targets to binary labels.
2. **Train/test split:** `train_models.py` creates a stratified 80:20 train/test split (`random_state=42`) and exports the untouched original-format test rows to `test_data.csv`.
3. **Numeric preprocessing:** Median imputation followed by standard scaling.
4. **Categorical preprocessing:** Most-frequent-value imputation followed by one-hot encoding (`handle_unknown="ignore"`).
5. **Leakage prevention:** Preprocessing is fitted within a separate scikit-learn `Pipeline` for each classifier using training data only.
6. **Evaluation:** Each trained Pipeline predicts on the same held-out test data. AUC uses class probabilities, or the SVM decision function; the other five metrics use predicted class labels.
7. **Saving models:** Five fitted Pipelines (including their preprocessing) are persisted as `.joblib` files in `model/`. The scores are saved to `model_comparison.csv`.
8. **Deployment:** `app.py` loads the saved Pipelines and evaluates labelled CSV data without retraining on the uploaded test file.

### Classifiers and configurations

| Classifier | Configuration in `train_models.py` |
|---|---|
| Logistic Regression | `max_iter=1500`, `class_weight="balanced"`, `random_state=42` |
| Decision Tree | `max_depth=7`, `min_samples_leaf=18`, `class_weight="balanced"`, `random_state=42` |
| K-Nearest Neighbors | `n_neighbors=15`, `weights="distance"` |
| Gaussian Naive Bayes | `var_smoothing=1e-8` |
| Support Vector Machine | RBF kernel, `C=1.0`, `gamma="scale"`, `class_weight="balanced"` |

### Streamlit application features

- Upload a labelled **CSV** containing all 17 original features and the `Revenue` target.
- Use the bundled `test_data.csv` as a demonstration test set when no file is uploaded.
- Compare the six evaluation metrics for all five saved classifiers in one table.
- Choose an individual model from a dropdown to inspect its metrics.
- View a confusion matrix, including true positives, true negatives, false positives and false negatives.
- Evaluate uploaded test data using the saved models **without fitting or retraining**.

## Reproduce in BITS Virtual Lab

Run these commands in the project directory on BITS Virtual Lab:

```bash
python -m pip install -r requirements.txt
python download_dataset.py
python train_models.py
streamlit run app.py
```

If the dataset download is blocked in the lab, obtain it from the official UCI link above and save the CSV as `data/online_shoppers_intention.csv`.

Once Streamlit starts, open the app, upload `test_data.csv`, compare all five models, and verify the confusion matrix for each model. Before final submission, confirm the public Streamlit link opens and that the GitHub README matches the README content included in the submitted PDF.

## Project files

| File / folder | Purpose |
|---|---|
| `app.py` | Interactive Streamlit interface for saved-model evaluation |
| `train_models.py` | Preprocessing, training, held-out test generation, model persistence and metrics |
| `common.py` | Dataset schema validation and the six shared metric calculations |
| `download_dataset.py` | Downloads the original UCI CSV into `data/` |
| `data/online_shoppers_intention.csv` | Source training dataset |
| `test_data.csv` | Held-out labelled test dataset used for evaluation |
| `model/` | Five trained `.joblib` Pipelines |
| `model_comparison.csv` | Results of all five models and six metrics |
| `requirements.txt` | Required Python dependencies |
| `submission_details.json` | Metadata, links and observations used by the submission generator |
| `prepare_submission.py` | Generates the combined submission PDF and updates README result sections |
| `ML_Assignment_Evedence.png` | Screenshot evidence file in the repository root |

**Expected model artifacts:** `logistic_regression.joblib`, `decision_tree.joblib`, `knn.joblib`, `naive_bayes.joblib`, and `support_vector_machine.joblib`.

## Dataset attribution

Sakar, C., and Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

Dataset access and use should follow the licence and attribution terms shown by UCI.
