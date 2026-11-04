# Machine Learning Assignment 2: Dry Bean Classification

## Problem statement

Identify the variety of a dry bean from its measured shape and size features. This project trains and compares five multiclass classifiers, evaluates them on the same test set, and provides an interactive Streamlit app for exploring model performance and individual predictions.

## Dataset description

- **Dataset:** [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), UCI Machine Learning Repository.
- **Size:** 13,611 records, 16 numeric input features, and seven target classes in the `Class` column: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.
- **License:** CC BY 4.0. [UCI dataset citation](https://doi.org/10.24432/C50S4B)
- **Evaluation split:** The dataset was split into ~ 80% training and 20% testing using a stratified, group-aware split. It exhibited a moderate class imbalance and exploratory analysis also revealed duplicate feature rows. So stratification was used to maintain similar class proportions across both sets, while grouping ensured that identical feature rows remained within the same partition in order to minimize duplicate related data leakage.

## GitHub repository

[Krishnakrish77/dry-bean-classifier](https://github.com/Krishnakrish77/dry-bean-classifier)

### Repository structure

```text
dry-bean-classifier/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── LICENSE
├── README.md
├── app.py
├── requirements.txt
├── test_data.csv
└── model/
    ├── data/
    │   ├── dry-bean-dataset.zip
    │   └── dry_bean.csv
    ├── decision_tree.joblib
    ├── gaussian_naive_bayes.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── support_vector_machine.joblib
    └── training.ipynb
```

## Models and evaluation

All five models use the same test set. Precision, recall, and F1 are macro-averaged; AUC is macro one-vs-rest. Values are rounded to four decimal places.

| ML model | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9273 | 0.9949 | 0.9393 | 0.9365 | 0.9378 | 0.9121 |
| Decision Tree | 0.8953 | 0.9447 | 0.9086 | 0.9077 | 0.9081 | 0.8734 |
| KNN | 0.9195 | 0.9841 | 0.9324 | 0.9308 | 0.9316 | 0.9027 |
| Gaussian Naive Bayes | 0.8942 | 0.9909 | 0.9044 | 0.9023 | 0.9024 | 0.8727 |
| Support Vector Machine | 0.9298 | 0.9908 | 0.9423 | 0.9388 | 0.9405 | 0.9151 |

The Dry Bean dataset contains seven classes with an uneven distribution. DERMASON is the majority class with 3,546 observations (26.1%), while BOMBAY is the minority class with 522 observations (3.8%).

So the model metrics, Precision, Recall, and F1 are evaluated using **macro averaging**, which calculates each metric separately for all seven classes and takes their unweighted average. This ensures that every bean variety contributes equally to the evaluation, regardless of its frequency in the dataset.

### Performance observations

| ML model | Observation |
|---|---|
| Logistic Regression | Strong, balanced results and the highest AUC (0.9949); it ranks second by macro F1. Very close to SVM. |
| Decision Tree | Lower accuracy and macro F1 than Logistic Regression, KNN, and SVM; it also has the lowest AUC (0.9447). |
| KNN | Consistently strong performance, with a macro F1 of 0.9316, but below Logistic Regression and SVM on this split. |
| Gaussian Naive Bayes | Has a high AUC (0.9909), but lower accuracy and macro F1. Its probability ranking is stronger than its final class predictions on this test set. |
| Support Vector Machine | Best accuracy, macro precision, macro recall, macro F1, and MCC. It has the strongest overall classification results by macro F1. |

**Overall winner: Support Vector Machine**, with the highest macro F1 (0.9405). Logistic Regression has the highest macro AUC (0.9949), so it performs best by that ranking metric. To be noted is that SVM had beaten logistic regression by a very thin margin, indicating that both models performed comparably on this test split.

## Streamlit app

- **Live app:** [dry-bean-classifier-ml.streamlit.app](https://dry-bean-classifier-ml.streamlit.app/)
- **Run locally:**

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  streamlit run app.py
  ```

The app loads the five saved models without retraining. It evaluates the bundled test data by default or accepts a labeled CSV up to 10 MB. The sidebar includes a downloadable CSV template and lists the required columns.

### Upload format

The uploaded CSV must contain these exact, case-sensitive headers:

```csv
Area,Perimeter,MajorAxisLength,MinorAxisLength,AspectRation,Eccentricity,ConvexArea,EquivDiameter,Extent,Solidity,roundness,Compactness,ShapeFactor1,ShapeFactor2,ShapeFactor3,ShapeFactor4,Class
```

Each feature column must contain numeric values. `Class` must contain one of `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, or `SIRA`. Add one labeled bean per row; the downloaded template contains only the header row. Column order may vary, and any additional columns are ignored.

The app also provides a model selector, evaluation metrics, confusion matrix, per-class classification report, and a prediction explorer.
