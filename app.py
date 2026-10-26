"""Streamlit demonstration: uploaded, labelled CSV -> five saved pipelines.

This file NEVER fits or retrains a model. Run:
    streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import (
    METRIC_NAMES, MODEL_DIR, MODEL_FILES, TARGET, TEST_FILE,
    score_model, validate_test_frame,
)

st.set_page_config(
    page_title="Shopping Intent | ML Classifier Lab",
    page_icon="🛒", layout="wide"
)

st.title("🛒 Online Shopping Intent — Classification Lab")
st.write(
    "Evaluate five previously trained classifiers for predicting whether a "
    "shopping session results in a purchase. No models are retrained in this app."
)
st.caption("Dataset: UCI Online Shoppers Purchasing Intention · Target: Revenue")


@st.cache_resource(show_spinner="Loading saved models...")
def load_pipeline(label):
    path = MODEL_DIR / MODEL_FILES[label]
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved model: {path.name}")
    return joblib.load(path)


st.subheader("1. Select labelled test data")
uploaded = st.file_uploader(
    "Upload a CSV containing the required 17 features AND Revenue target",
    type=["csv"], accept_multiple_files=False,
)
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.info("Using your uploaded test CSV.")
elif TEST_FILE.exists():
    raw = pd.read_csv(TEST_FILE)
    st.info("Using the bundled held-out test_data.csv. You can upload another test CSV above.")
else:
    st.warning("Upload test_data.csv to begin. No dataset is bundled yet.")
    st.stop()

try:
    X_test, y_test = validate_test_frame(raw)
except Exception as exc:
    st.error(f"Invalid CSV: {exc}")
    st.stop()

if len(set(y_test)) < 2:
    st.warning("Only one class found. ROC AUC is undefined; use a two-class test CSV.")

c1, c2, c3 = st.columns(3)
c1.metric("Test rows", f"{len(raw):,}")
c2.metric("Predictor features", X_test.shape[1])
c3.metric("Purchasing sessions", f"{sum(y_test):,}")
with st.expander("Preview uploaded/bundled dataset"):
    st.dataframe(raw.head(12), use_container_width=True)

st.subheader("2. Compare all five models")
results = []
loaded_models = {}
for label in MODEL_FILES:
    try:
        model = load_pipeline(label)
        loaded_models[label] = model
        results.append({"ML Model Name": label, **score_model(model, X_test, y_test)})
    except Exception as exc:
        st.error(f"Unable to evaluate {label}: {exc}")
        st.stop()
comparison = pd.DataFrame(results).set_index("ML Model Name")
st.dataframe(comparison.style.format({x: "{:.4f}" for x in METRIC_NAMES}), use_container_width=True)
st.caption("AUC uses probability scores or the SVM decision function; all other metrics use predicted labels.")

st.subheader("3. Explore one model")
selected = st.selectbox("Choose an ML classifier", options=list(MODEL_FILES))
selected_model = loaded_models[selected]
selected_metrics = comparison.loc[selected]
metric_columns = st.columns(6)
for col, metric_name in zip(metric_columns, METRIC_NAMES):
    col.metric(metric_name, f"{selected_metrics[metric_name]:.3f}")

predictions = selected_model.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
left, right = st.columns([1, 1])
with left:
    st.write("**Confusion matrix** (actual × predicted)")
    figure, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No purchase (0)", "Purchase (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.setp(ax.get_xticklabels(), rotation=13, ha="right")
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
with right:
    st.write("**Interpret the outcome**")
    st.write(f"True negatives: {cm[0,0]:,}")
    st.write(f"False positives: {cm[0,1]:,}")
    st.write(f"False negatives: {cm[1,0]:,}")
    st.write(f"True positives: {cm[1,1]:,}")
    st.write(
        "**Study question:** Does this model favor identifying purchasers "
        "(Recall) or avoiding false purchase predictions (Precision)?"
    )

st.divider()
st.caption(
    "Educational evaluation interface. Always use test labels for metrics, "
    "and never fit or retrain models on evaluation data."
)
