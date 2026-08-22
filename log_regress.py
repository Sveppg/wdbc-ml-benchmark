import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. Spaltennamen des WDBC-Datensatzes
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Datensatz laden
# ------------------------------------------------------------

df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=columns
)

print(df.head())
print("\nDimension:", df.shape)

print("\nKlassenverteilung:")
print(df["diagnosis"].value_counts())


# ------------------------------------------------------------
# 3. Features und Zielvariable
# ------------------------------------------------------------

# ID hat keine diagnostische Bedeutung und wird entfernt.
X = df.drop(columns=["id", "diagnosis"])

# malignant = 1
# benign    = 0
y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# ------------------------------------------------------------
# 4. Train-Test-Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 5. Logistic Regression + Standardisierung
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),

    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ------------------------------------------------------------
# 6. Trainingszeit
# ------------------------------------------------------------

start = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start


# ------------------------------------------------------------
# 7. Vorhersagezeit
# ------------------------------------------------------------

start = time.perf_counter()

y_pred = model.predict(X_test)

prediction_time = time.perf_counter() - start


# ------------------------------------------------------------
# 8. Metriken
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# ------------------------------------------------------------
# 9. Ergebnisse
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print(f"Accuracy:              {accuracy:.4f}")
print(f"Precision malignant:   {precision:.4f}")
print(f"Recall malignant:      {recall:.4f}")
print(f"F1-Score malignant:    {f1:.4f}")

print(f"\nTrainingszeit:         {training_time:.6f} Sekunden")
print(f"Vorhersagezeit:        {prediction_time:.6f} Sekunden")


# ------------------------------------------------------------
# 10. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 11. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["benign", "malignant"]
    )
)

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
            accuracy_score,
                precision_score,
                    recall_score,
                        f1_score,
                            confusion_matrix,
                                classification_report
                                )


# ------------------------------------------------------------
# 1. Spaltennamen des WDBC-Datensatzes
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Datensatz laden
# ------------------------------------------------------------

df = pd.read_csv(
            "wdbc.data",
                header=None,
                    names=columns
                    )

print(df.head())
print("\nDimension:", df.shape)

print("\nKlassenverteilung:")
print(df["diagnosis"].value_counts())


# ------------------------------------------------------------
# 3. Features und Zielvariable
# ------------------------------------------------------------

# ID hat keine diagnostische Bedeutung und wird entfernt.
X = df.drop(columns=["id", "diagnosis"])

# malignant = 1
# benign    = 0
y = df["diagnosis"].map({
        "M": 1,
            "B": 0
            })


# ------------------------------------------------------------
# 4. Train-Test-Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
            X,
                y,
                    test_size=0.2,
                        random_state=42,
                            stratify=y
                            )


# ------------------------------------------------------------
# 5. Logistic Regression + Standardisierung
# ------------------------------------------------------------

model = Pipeline([
        ("scaler", StandardScaler()),

            ("logistic_regression", LogisticRegression(
                        max_iter=1000,
                                random_state=42
                                    ))
            ])


# ------------------------------------------------------------
# 6. Trainingszeit
# ------------------------------------------------------------

start = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start


# ------------------------------------------------------------
# 7. Vorhersagezeit
# ------------------------------------------------------------

start = time.perf_counter()

y_pred = model.predict(X_test)

prediction_time = time.perf_counter() - start


# ------------------------------------------------------------
# 8. Metriken
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# ------------------------------------------------------------
# 9. Ergebnisse
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print(f"Accuracy:              {accuracy:.4f}")
print(f"Precision malignant:   {precision:.4f}")
print(f"Recall malignant:      {recall:.4f}")
print(f"F1-Score malignant:    {f1:.4f}")

print(f"\nTrainingszeit:         {training_time:.6f} Sekunden")
print(f"Vorhersagezeit:        {prediction_time:.6f} Sekunden")


# ------------------------------------------------------------
# 10. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 11. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
            classification_report(
                        y_test,
                                y_pred,
                                        target_names=["benign", "malignant"]
                                            )
            )
