from pathlib import Path

import altair as alt
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

with st.container(border=True):
    st.badge("UCI DRY BEAN · MODEL EVALUATION", color="green")
    st.title("BeanLab 🫘")
    st.write("Compare and explore how five different machine learning models classify dry bean varieties using the bean's geometric features.")

missing_models = [name for name, path in MODEL_PATHS.items() if not path.is_file()]
if missing_models:
    fail("Saved model files are missing.", ", ".join(missing_models))

try:
    sample_model = load_model(str(next(iter(MODEL_PATHS.values()))))
    feature_columns = list(sample_model.feature_names_in_)
    class_names = list(sample_model.classes_)
except Exception as error:
    fail("Could not load the saved model metadata.", error)

required_columns = [*feature_columns, TARGET_COLUMN]
csv_template = pd.DataFrame(columns=required_columns).to_csv(index=False).encode("utf-8")

with st.sidebar:
    st.subheader("Evaluation setup")
    uploaded_file = st.file_uploader(
        "Labeled test data (CSV)",
        type=["csv"],
        help="CSV must contain all 16 Dry Bean features and a Class column.",
    )
    st.download_button(
        "Download CSV template",
        data=csv_template,
        file_name="dry_bean_test_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.expander("Required CSV columns"):
        st.caption("Headers are case-sensitive. Add one labeled bean record per row.")
        st.code(",".join(required_columns), language="text")
        st.caption(
            "All 16 feature values must be numeric. Class must be one of: "
            + ", ".join(class_names)
            + "."
        )
    selected_model_name = st.selectbox("Model to inspect", list(MODEL_PATHS))
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
    data_source = "Bundled test split"
else:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        fail(f"The uploaded file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Upload only the test split.")
    try:
        evaluation_data = pd.read_csv(uploaded_file)
    except Exception as error:
        fail("Could not read the uploaded CSV.", error)
    data_source = f"Uploaded · {uploaded_file.name}"

st.badge(
    f"Evaluation source · {data_source}",
    color="blue" if uploaded_file is not None else "green",
)

if evaluation_data.empty:
    fail("The CSV contains no data rows.")

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

overview_tab, comparison_tab, explorer_tab = st.tabs(
    ["Overview", "Model Comparison", "Bean Explorer"]
)

with overview_tab:
    class_counts = y_eval.value_counts().reindex(class_names, fill_value=0)
    st.subheader("Evaluation overview")
    summary_columns = st.columns(4)
    summary_metrics = [
        ("Rows evaluated", f"{len(evaluation_data):,}"),
        ("Input features", f"{len(feature_columns)}"),
        ("Bean varieties", f"{len(observed_class_names)} / {len(class_names)}"),
        ("Models evaluated", f"{len(evaluation_results)}"),
    ]
    for column, (label, value) in zip(summary_columns, summary_metrics):
        with column.container(border=True):
            st.metric(label, value)

    st.markdown("#### Class distribution")
    st.caption("Number of evaluated records in each bean variety.")
    class_distribution = pd.DataFrame(
        {"Class": class_counts.index, "Records": class_counts.to_numpy()}
    )
    distribution_chart = (
        alt.Chart(class_distribution)
        .mark_bar(color="#28735f", cornerRadiusEnd=5)
        .encode(
            x=alt.X("Records:Q", title="Evaluated records"),
            y=alt.Y("Class:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Class:N"),
                alt.Tooltip("Records:Q", format=","),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(distribution_chart, width="stretch")

with comparison_tab:
    best_row = comparison.iloc[0]
    st.markdown("#### Best model by macro F1")
    best_left, best_middle, best_right = st.columns(3)
    with best_left.container(border=True):
        st.caption("Top-performing model")
        st.markdown(f"#### {best_row['Model']}")
    with best_middle.container(border=True):
        st.metric("Macro F1", f"{best_row['F1']:.3f}")
    with best_right.container(border=True):
        st.metric("MCC", f"{best_row['MCC']:.3f}")
    st.caption("Highest macro F1 on the evaluated data; MCC is shown as a cross-check.")

    chart_metric = st.selectbox(
        "Metric to visualize",
        ["F1", "Accuracy", "AUC", "Precision", "Recall", "MCC"],
        format_func=lambda metric: {
            "F1": "Macro F1",
            "Accuracy": "Accuracy",
            "AUC": "Macro AUC",
            "Precision": "Macro precision",
            "Recall": "Macro recall",
            "MCC": "Matthews correlation coefficient",
        }[metric],
    )
    f1_chart = (
        alt.Chart(comparison)
        .mark_bar(color="#28735f", cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                f"{chart_metric}:Q",
                title=chart_metric,
                scale=alt.Scale(domain=[-1, 1] if chart_metric == "MCC" else [0, 1]),
            ),
            y=alt.Y(
                "Model:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=180),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(f"{chart_metric}:Q", format=".3f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(f1_chart, width="stretch")

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

with explorer_tab:
    st.markdown("#### Inspect an evaluated record")
    st.caption("Choose an actual variety and record to compare its prediction and confidence.")
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
    with sample_left:
        with st.container(border=True):
            actual_class = y_eval.iloc[selected_position]
            is_correct = sample_prediction == actual_class
            st.badge(
                "Correct prediction" if is_correct else "Misclassified",
                color="green" if is_correct else "red",
            )
            st.metric("Predicted variety", str(sample_prediction))
            st.caption(f"Actual variety · {actual_class}")
            st.metric("Prediction confidence", f"{sample_probabilities.max():.1%}")
    with sample_right:
        probability_frame = pd.DataFrame(
            {
                "Variety": selected_model.classes_,
                "Probability": sample_probabilities,
            }
        )
        probability_chart = (
            alt.Chart(probability_frame)
            .mark_bar(color="#28735f", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Probability:Q", title="Prediction probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Variety:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Variety:N"),
                    alt.Tooltip("Probability:Q", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(probability_chart, width="stretch")
    st.markdown("#### Measured features")
    feature_values = (
        sample_features.iloc[0]
        .rename("Value")
        .rename_axis("Feature")
        .reset_index()
    )
    st.dataframe(
        feature_values,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Value": st.column_config.NumberColumn(
                "Value", width="small", alignment="left"
            ),
        },
        width=420,
        hide_index=True,
    )