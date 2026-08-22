import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# 1. WDBC-Datensatz laden
# ============================================================

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


df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=columns
)


# ============================================================
# 2. Features und Zielvariable
# ============================================================

X = df.drop(columns=["id", "diagnosis"])

# malignant = 1
# benign    = 0
y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# ============================================================
# 3. Train-Test-Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. Modelle
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Gaussian Naive Bayes": GaussianNB(),

    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


# ============================================================
# 5. Funktion zur Modellbewertung
# ============================================================

def evaluate_model(name, model):

    # --------------------
    # Training
    # --------------------

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start


    # --------------------
    # Prediction
    # --------------------

    start = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start


    # --------------------
    # Metriken
    # --------------------

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Training Time": training_time,
        "Prediction Time": prediction_time
    }


    # --------------------
    # Ausgabe
    # --------------------

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print(f"Accuracy:              {metrics['Accuracy']:.4f}")
    print(f"Precision malignant:   {metrics['Precision']:.4f}")
    print(f"Recall malignant:      {metrics['Recall']:.4f}")
    print(f"F1-Score malignant:    {metrics['F1']:.4f}")

    print(
        f"\nTrainingszeit:         "
        f"{metrics['Training Time']:.6f} Sekunden"
    )

    print(
        f"Vorhersagezeit:        "
        f"{metrics['Prediction Time']:.6f} Sekunden"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    return metrics, y_pred


# ============================================================
# 6. Alle Modelle ausführen
# ============================================================

results = {}
predictions = {}


for name, model in models.items():

    metrics, y_pred = evaluate_model(
        name,
        model
    )

    results[name] = metrics
    predictions[name] = y_pred


# ============================================================
# 7. Confusion Matrices
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)


for ax, (name, y_pred) in zip(
    axes,
    predictions.items()
):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Benign",
            "Malignant"
        ],
        ax=ax,
        colorbar=False
    )

    ax.set_title(name)


fig.suptitle(
    "Confusion Matrices"
)

plt.tight_layout()

# Optional für dein Paper:
# plt.savefig(
#     "confusion_matrices.png",
#     dpi=300
# )

plt.show()


# ============================================================
# 8. ROC-Kurven
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for name, model in models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )


# Zufallsklassifikator
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_title(
    "ROC Curve Comparison"
)

ax.legend()

plt.tight_layout()

# Optional:
# plt.savefig(
#     "roc_curves.png",
#     dpi=300
# )

plt.show()


# ============================================================
# 9. Performance-Metriken vergleichen
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]


x = np.arange(
    len(metric_names)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(10, 6)
)


for index, (name, metrics) in enumerate(
    results.items()
):

    values = [
        metrics[metric]
        for metric in metric_names
    ]

    position = (
        x
        + (index - 1) * width
    )

    ax.bar(
        position,
        values,
        width,
        label=name
    )


ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Model Performance Comparison"
)

ax.set_xticks(x)

ax.set_xticklabels(
    metric_names
)

ax.set_ylim(
    0,
    1.05
)

ax.legend()

plt.tight_layout()

# Optional:
# plt.savefig(
#     "performance_comparison.png",
#     dpi=300
# )

plt.show()


# ============================================================
# 10. Laufzeiten vergleichen
# ============================================================

model_names = list(
    results.keys()
)


training_times = [
    results[name]["Training Time"]
    for name in model_names
]


prediction_times = [
    results[name]["Prediction Time"]
    for name in model_names
]


x = np.arange(
    len(model_names)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)


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


ax.set_ylabel(
    "Time in seconds"
)

ax.set_title(
    "Computational Performance"
)

ax.set_xticks(x)

ax.set_xticklabels(
    model_names
)

ax.legend()

plt.tight_layout()

# Optional:
# plt.savefig(
#     "runtime_comparison.png",
#     dpi=300
# )

plt.show()