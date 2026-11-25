
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# --------------------------------------------------
# Dataset Loading and Preprocessing
# --------------------------------------------------

# Column names according to the original WDBC dataset
columns = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# Load the dataset without an existing header
df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=columns
)

# Extract the 30 numerical input features
X = df.drop(columns=["id", "diagnosis"])

# Encode the binary target variable
# Malignant = 1 (positive class)
# Benign    = 0 (negative class)
y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

# Stratified 80/20 split to preserve class distribution
# A fixed random seed improves reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Machine Learning Models
# --------------------------------------------------

# Pipelines apply feature scaling during model training
# and prevent data leakage from the test dataset
models = {

    # Linear classification model
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    # Probabilistic classifier based on Bayes' theorem
    "Gaussian Naive Bayes": GaussianNB(),

    # Distance-based classification using five neighbors
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    # Support Vector Machine with default RBF kernel
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=False,
            random_state=42
        ))
    ]),

    # Ensemble of 100 decision trees
    # Use all available CPU cores where supported
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

def evaluate_model(name, model):
    """Train and evaluate a classification model."""

    # Measure model training time
    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    # Measure prediction time on the test dataset
    start = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start

    # Calculate classification and computational metrics
    # Precision, recall and F1 refer to the malignant class
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Training Time": training_time,
        "Prediction Time": prediction_time
    }

    # Display evaluation results in the console
    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print(f"Accuracy:              {metrics['Accuracy']:.4f}")
    print(f"Precision malignant:   {metrics['Precision']:.4f}")
    print(f"Recall malignant:      {metrics['Recall']:.4f}")
    print(f"F1-Score malignant:    {metrics['F1']:.4f}")

    print(
        f"\nTraining Time:         "
        f"{metrics['Training Time']:.6f} seconds"
    )

    print(
        f"Prediction Time:       "
        f"{metrics['Prediction Time']:.6f} seconds"
    )

    # Rows represent actual classes, columns predicted classes
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics, y_pred


# Store evaluation metrics and predictions for each model
results = {}
predictions = {}

# Train and evaluate all defined models
for name, model in models.items():

    metrics, y_pred = evaluate_model(name, model)

    results[name] = metrics
    predictions[name] = y_pred


# --------------------------------------------------
# Confusion Matrices
# --------------------------------------------------

number_of_models = len(models)

# Create one subplot for each classification model
fig, axes = plt.subplots(
    1,
    number_of_models,
    figsize=(5 * number_of_models, 4)
)

# Visualize correct and incorrect classifications
for ax, (name, y_pred) in zip(axes, predictions.items()):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Benign", "Malignant"],
        ax=ax,
        colorbar=False
    )

    ax.set_title(name)

fig.suptitle("Confusion Matrices")

plt.tight_layout()

# Optional: export the figure as a high-resolution image
# plt.savefig(
#     "confusion_matrices.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()


# --------------------------------------------------
# ROC Curve and AUC Comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

# Plot ROC curves and AUC scores for all trained models
# SVM uses decision scores instead of class probabilities
for name, model in models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

# Add a random classifier as a reference baseline
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_title("ROC Curve Comparison")
ax.legend()

plt.tight_layout()

# Optional: export the ROC comparison
# plt.savefig(
#     "roc_curves.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()


# --------------------------------------------------
# Classification Performance Comparison
# --------------------------------------------------

# Select the classification metrics to be visualized
metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

x = np.arange(len(metric_names))

number_of_models = len(models)

# Calculate the bar width for grouped visualization
width = 0.8 / number_of_models

fig, ax = plt.subplots(figsize=(11, 6))

# Plot classification metrics for each model
for index, (name, metrics) in enumerate(results.items()):

    values = [
        metrics[metric]
        for metric in metric_names
    ]

    # Center the grouped bars around each metric
    offset = (
        index - (number_of_models - 1) / 2
    ) * width

    ax.bar(
        x + offset,
        values,
        width,
        label=name
    )

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")

ax.set_xticks(x)
ax.set_xticklabels(metric_names)

# Classification scores range from 0 to 1
ax.set_ylim(0, 1.05)

ax.legend()

plt.tight_layout()

# Optional: export the classification comparison
# plt.savefig(
#     "performance_comparison.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()


# --------------------------------------------------
# Computational Performance Comparison
# --------------------------------------------------

# Extract measured execution times for each model
model_names = list(results.keys())

training_times = [
    results[name]["Training Time"]
    for name in model_names
]

prediction_times = [
    results[name]["Prediction Time"]
    for name in model_names
]

x = np.arange(len(model_names))

width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

# Plot training and prediction times side by side
ax.bar(
    x - width / 2,
    training_times,
    width,
    label="Training Time"
)

ax.bar(
    x + width / 2,
    prediction_times,
    width,
    label="Prediction Time"
)

ax.set_ylabel("Time in seconds")
ax.set_title("Computational Performance")

ax.set_xticks(x)
ax.set_xticklabels(model_names)

ax.legend()

plt.tight_layout()

# Optional: export the computational comparison
# plt.savefig(
#     "runtime_comparison.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()
