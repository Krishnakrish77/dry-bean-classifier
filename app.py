from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "Class"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "model"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_PATHS = {
    "Logistic Regression": MODEL_DIRECTORY / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIRECTORY / "decision_tree.joblib",
    "KNN": MODEL_DIRECTORY / "knn.joblib",
    "Gaussian Naive Bayes": MODEL_DIRECTORY / "gaussian_naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIRECTORY / "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="BeanLab | Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

def fail(message: str, details: object = None) -> None:
    """Show a fixed error message, put any dynamic detail in plain-text code, then stop."""
    st.error(message)
    if details is not None:
        st.code(str(details), language=None)
    st.stop()

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def calculate_metrics(model, features: pd.DataFrame, targets: pd.Series) -> dict:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = model.classes_

    auc = np.nan
    if set(pd.unique(targets)) == set(classes):
        auc = roc_auc_score(
            targets,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )

    return {
        "Accuracy": accuracy_score(targets, predictions),
        "AUC": auc,
        "Precision": precision_score(
            targets, predictions, labels=classes, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            targets, predictions, labels=classes, average="macro", zero_division=0
        ),
        "F1": f1_score(
            targets, predictions, labels=classes, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(targets, predictions),
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": classes,
    }

st.title("BeanLab")
st.caption("UCI Dry Bean · Model Evaluation")

missing_models = [name for name, path in MODEL_PATHS.items() if not path.is_file()]
if missing_models:
    fail("Saved model files are missing.", ", ".join(missing_models))

try:
    sample_model = load_model(str(next(iter(MODEL_PATHS.values()))))
    feature_columns = list(sample_model.feature_names_in_)
    class_names = list(sample_model.classes_)
except Exception as error:
    fail("Could not load the saved model metadata.", error)

with st.sidebar:
    st.subheader("Evaluation setup")
    uploaded_file = st.file_uploader(
        "Labeled test data (CSV)",
        type=["csv"],
        help="CSV must contain all 16 Dry Bean features and a Class column.",
    )
    selected_model_name = st.selectbox("Model detail", list(MODEL_PATHS))
    st.caption("The app loads saved models and evaluates the uploaded data; it does not retrain.")
    with st.expander("Dataset citation"):
        st.markdown(
            "Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. "
            "[https://doi.org/10.24432/C50S4B](https://doi.org/10.24432/C50S4B)\n\n"
            "Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)."
        )

# Validations
if uploaded_file is None:
    if not TEST_DATA_PATH.is_file():
        fail("Upload a labeled CSV; the bundled test_data.csv was not found.")
    evaluation_data = pd.read_csv(TEST_DATA_PATH)
    st.caption("Showing the bundled test data. Upload another labeled CSV to evaluate it.")
else:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        fail(f"The uploaded file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Upload only the test split.")
    try:
        evaluation_data = pd.read_csv(uploaded_file)
    except Exception as error:
        fail("Could not read the uploaded CSV.", error)

if evaluation_data.empty:
    fail("The CSV contains no data rows.")

required_columns = [*feature_columns, TARGET_COLUMN]
missing_columns = [column for column in required_columns if column not in evaluation_data.columns]
if missing_columns:
    fail("Required columns are missing.", ", ".join(map(str, missing_columns)))

evaluation_data = evaluation_data.loc[:, required_columns].copy()
evaluation_data[feature_columns] = evaluation_data[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)
if evaluation_data[required_columns].isna().any().any():
    fail("The CSV contains blank or non-numeric feature/target values.")

observed_classes = set(evaluation_data[TARGET_COLUMN].unique())
unknown_classes = observed_classes.difference(class_names)
if unknown_classes:
    fail("Unknown Class values.", ", ".join(map(str, sorted(unknown_classes))))

observed_class_names = [name for name in class_names if name in observed_classes]

X_eval = evaluation_data[feature_columns]
y_eval = evaluation_data[TARGET_COLUMN]

evaluation_results = {}
comparison_rows = []
try:
    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(str(model_path))
        result = calculate_metrics(model, X_eval, y_eval)
        evaluation_results[model_name] = result
        comparison_rows.append({"Model": model_name, **{key: result[key] for key in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}})
except Exception as error:
    fail("Model evaluation failed.", error)

comparison = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False)
selected_result = evaluation_results[selected_model_name]
selected_model = load_model(str(MODEL_PATHS[selected_model_name]))

summary_left, summary_right = st.columns([1, 3])
with summary_left:
    st.metric("Rows evaluated", f"{len(evaluation_data):,}")
    st.metric("Input features", len(feature_columns))
    observed_counts = y_eval.value_counts()

with summary_right:
    class_counts = y_eval.value_counts().reindex(class_names, fill_value=0)
    st.bar_chart(class_counts.rename("Rows"), height=250, color="#28735f")

st.subheader("All-model comparison")
st.dataframe(
    comparison.style.format(
        {metric: "{:.3f}" for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
    ),
    width="stretch",
    hide_index=True,
)
st.caption("Precision, recall, and F1 use macro averaging; AUC is macro one-vs-rest.")

st.subheader(selected_model_name)
metric_columns = st.columns(6)
for column, metric in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = selected_result[metric]
    column.metric(metric, "N/A" if pd.isna(value) else f"{value:.3f}")

detail_left, detail_right = st.columns([1.15, 1])
with detail_left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_eval, selected_result["predictions"], labels=class_names)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with detail_right:
    st.markdown("#### Per-class report")
    report = classification_report(
        y_eval,
        selected_result["predictions"],
        labels=class_names,
        target_names=[str(name) for name in class_names],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T
    st.dataframe(
        report_frame.loc[
            [*map(str, class_names), "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"],
        ].style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}),
        width="stretch",
    )

with st.expander("Inspect one evaluated record"):
    actual_class = st.selectbox("Actual bean class", observed_class_names)
    class_positions = np.flatnonzero(y_eval.to_numpy() == actual_class).tolist()
    selected_position = st.selectbox(
        "Test example",
        options=class_positions,
        format_func=lambda position: (
            f"Example {position + 1} · Area {X_eval.iloc[position]['Area']:,.0f} · "
            f"Perimeter {X_eval.iloc[position]['Perimeter']:,.1f} · "
            f"Major axis {X_eval.iloc[position]['MajorAxisLength']:,.1f}"
        ),
    )
    sample_features = X_eval.iloc[[selected_position]]
    sample_prediction = selected_model.predict(sample_features)[0]
    sample_probabilities = selected_model.predict_proba(sample_features)[0]
    sample_left, sample_right = st.columns([1, 2])
    sample_left.metric("Predicted variety", str(sample_prediction))
    sample_left.text(f"Actual variety: {y_eval.iloc[selected_position]}")
    sample_left.metric("Prediction confidence", f"{sample_probabilities.max():.1%}")
    sample_right.bar_chart(
        pd.Series(sample_probabilities, index=selected_model.classes_, name="Probability"),
        color="#e6a544",
    )
    st.markdown("#### Measured features")
    feature_values = sample_features.iloc[0].rename("Value").to_frame()
    feature_values.index.name = "Feature"
    st.dataframe(feature_values, width="stretch")