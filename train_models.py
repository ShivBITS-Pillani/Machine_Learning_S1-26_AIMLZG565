"""Fit the five required classifiers using ONLY the training split.

Run in BITS Virtual Lab:
    python train_models.py

Outputs:
    model/*.joblib     -> one saved Pipeline per classifier
    test_data.csv      -> untouched 20% hold-out test data incl. Revenue
    model_comparison.csv -> results on that held-out data

Explore and justify your own hyperparameters in your assignment reflections.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (
    CATEGORICAL_FEATURES, DATA_FILE, FEATURES, METRICS_FILE, MODEL_DIR,
    MODEL_FILES, NUMERIC_FEATURES, TARGET, TEST_FILE, score_model,
    validate_test_frame,
)

# Adjust these decisions only after you understand them. Describe your
# chosen test split and hyperparameters in your own README observations.
SPLIT_SEED = 42
TEST_FRACTION = 0.20


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numerical", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", category_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")


def classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=SPLIT_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=18,
            class_weight="balanced", random_state=SPLIT_SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"
        ),
    }


def main() -> None:
    if not DATA_FILE.is_file():
        raise SystemExit(
            f"Missing {DATA_FILE.name}. First run 'python download_dataset.py' "
            "or download the UCI CSV manually into data/."
        )

    original = pd.read_csv(DATA_FILE)
    X, y = validate_test_frame(original)
    if len(X) < 500 or len(FEATURES) < 12:
        raise ValueError("Dataset does not meet assignment's minimum size")
    if len(set(y)) != 2:
        raise ValueError("Binary dataset must have two target classes")
    print(f"Dataset: {len(X)} rows | {len(FEATURES)} source features | target=Revenue")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SPLIT_SEED, stratify=y
    )

    # Serialize the held-out ORIGINAL rows; not the transformed columns.
    # A reviewer can upload this same test_data.csv to Streamlit, and every
    # saved Pipeline will automatically preprocess it exactly as during train.
    _, original_test = train_test_split(
        original[FEATURES + [TARGET]], test_size=TEST_FRACTION,
        random_state=SPLIT_SEED, stratify=y
    )
    original_test.reset_index(drop=True).to_csv(TEST_FILE, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, estimator in classifiers().items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", estimator),
        ])
        print("Training:", label, flush=True)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / MODEL_FILES[label])
        scores = score_model(pipeline, X_test, y_test)
        rows.append({"ML Model Name": label, **scores})
        print("  ", ", ".join(f"{name}: {value:.4f}" for name, value in scores.items()))

    results = pd.DataFrame(rows)
    results.to_csv(METRICS_FILE, index=False)
    print("\nMODEL COMPARISON (SAVED TO CSV)")
    print(results.round(4).to_string(index=False))
    print(f"\nTest data: {TEST_FILE.name} ({len(X_test)} rows)")
    print(f"Saved models: {MODEL_DIR} ({len(rows)} files)")
    print("Next: run 'streamlit run app.py' on BITS Virtual Lab")


if __name__ == "__main__":
    main()
