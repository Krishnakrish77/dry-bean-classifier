# Machine Learning Assignment 2: Dry Bean Classification

## Problem statement

Identify the variety of a dry bean from its measured shape and size features. This project trains and compares five multiclass classifiers, evaluates them on the same test set, and provides an interactive Streamlit app for exploring model performance and individual predictions.

## Dataset description

- **Dataset:** [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), UCI Machine Learning Repository.
- **Size:** 13,611 records, 16 numeric input features, and seven target classes in the `Class` column: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.
- **License:** CC BY 4.0. [UCI dataset citation](https://doi.org/10.24432/C50S4B)
- **Evaluation split:** A shuffled, stratified five-fold group split with `random_state=42`; the first fold is created for testing (10,889 training records and 2,722 test records). Identical features are kept in the same group to prevent duplicates across the train/test splits.

## GitHub repository

[Krishnakrish77/dry-bean-classifier](https://github.com/Krishnakrish77/dry-bean-classifier)

The repository includes `app.py`, `requirements.txt`, the test split `test_data.csv`, five saved model pipelines in `model/`, and the `model/training.ipynb` notebook.

## Models and evaluation

All five models use the same test set. Precision, recall, and F1 are macro-averaged; AUC is macro one-vs-rest. Values are rounded to four decimal places.

| ML model | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9273 | 0.9949 | 0.9393 | 0.9365 | 0.9378 | 0.9121 |
| Decision Tree | 0.8953 | 0.9447 | 0.9086 | 0.9077 | 0.9081 | 0.8734 |
| KNN | 0.9195 | 0.9841 | 0.9324 | 0.9308 | 0.9316 | 0.9027 |
| Gaussian Naive Bayes | 0.8942 | 0.9909 | 0.9044 | 0.9023 | 0.9024 | 0.8727 |
| Support Vector Machine | 0.9298 | 0.9908 | 0.9423 | 0.9388 | 0.9405 | 0.9151 |

### Performance observations

| ML model | Observation |
|---|---|
| Logistic Regression | Strong, balanced results and the highest AUC (0.9949); it ranks second by macro F1. Very close to SVM. |
| Decision Tree | Lower accuracy and macro F1 than Logistic Regression, KNN, and SVM; it also has the lowest AUC (0.9447). |
| KNN | Consistently strong performance, with a macro F1 of 0.9316, but below Logistic Regression and SVM on this split. |
| Gaussian Naive Bayes | Has a high AUC (0.9909), but lower accuracy and macro F1. Its probability ranking is stronger than its final class predictions on this test set. |
| Support Vector Machine | Best accuracy, macro precision, macro recall, macro F1, and MCC. It has the strongest overall classification results by macro F1. |

**Overall winner: Support Vector Machine**, with the highest macro F1 (0.9405). Logistic Regression has the highest macro AUC (0.9949), so it performs best by that ranking metric.

## Streamlit app

- **Live app:** [dry-bean-classifier-ml.streamlit.app](https://dry-bean-classifier-ml.streamlit.app/)
- **Run locally:**

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  streamlit run app.py
  ```

The app loads the five saved models without retraining. It evaluates the bundled test data by default or accepts a labeled CSV up to 10 MB containing all 16 features and the `Class` target. It provides model selection, evaluation metrics, a confusion matrix, a per-class report, a record-level prediction explorer.
