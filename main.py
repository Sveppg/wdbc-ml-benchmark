import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

# ID und Diagnose sind keine Features
X = df.drop(columns=["id", "diagnosis"])

# 1 = malignant
# 0 = benign
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
# 4. Modelle definieren
# ============================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


bayes_model = GaussianNB()


# ============================================================
# 5. Modelle trainieren
# ============================================================

logistic_model.fit(X_train, y_train)
bayes_model.fit(X_train, y_train)


# ============================================================
# 6. Vorhersagen
# ============================================================

y_pred_logistic = logistic_model.predict(X_test)
y_pred_bayes = bayes_model.predict(X_test)


# ============================================================
# 7. Metriken berechnen
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }


logistic_metrics = calculate_metrics(
    y_test,
    y_pred_logistic
)

bayes_metrics = calculate_metrics(
    y_test,
    y_pred_bayes
)


# ============================================================
# 8. Ergebnisse ausgeben
# ============================================================

print("\nLOGISTIC REGRESSION")
print("=" * 40)

for metric, value in logistic_metrics.items():
    print(f"{metric}: {value:.4f}")


print("\nGAUSSIAN NAIVE BAYES")
print("=" * 40)

for metric, value in bayes_metrics.items():
    print(f"{metric}: {value:.4f}")


# ============================================================
# 9. Confusion Matrix - Logistic Regression
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_logistic,
    display_labels=["Benign", "Malignant"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()


# ============================================================
# 10. Confusion Matrix - Gaussian Naive Bayes
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_bayes,
    display_labels=["Benign", "Malignant"]
)

plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.tight_layout()
plt.show()


# ============================================================
# 11. ROC-Kurven vergleichen
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))


RocCurveDisplay.from_estimator(
    logistic_model,
    X_test,
    y_test,
    name="Logistic Regression",
    ax=ax
)


RocCurveDisplay.from_estimator(
    bayes_model,
    X_test,
    y_test,
    name="Gaussian Naive Bayes",
    ax=ax
)


# Zufallsklassifikator
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_title("ROC Curve Comparison")
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 12. Vergleich der Metriken
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]


logistic_values = [
    logistic_metrics[metric]
    for metric in metric_names
]


bayes_values = [
    bayes_metrics[metric]
    for metric in metric_names
]


x = np.arange(len(metric_names))

width = 0.35


fig, ax = plt.subplots(figsize=(9, 5))


ax.bar(
    x - width / 2,
    logistic_values,
    width,
    label="Logistic Regression"
)


ax.bar(
    x + width / 2,
    bayes_values,
    width,
    label="Gaussian Naive Bayes"
)


ax.set_ylabel("Score")

ax.set_title(
    "Performance Comparison"
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
            accuracy_score,
                precision_score,
                    recall_score,
                        f1_score,
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

# ID und Diagnose sind keine Features
X = df.drop(columns=["id", "diagnosis"])

# 1 = malignant
# 0 = benign
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
# 4. Modelle definieren
# ============================================================

logistic_model = Pipeline([
        ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                        max_iter=1000,
                                random_state=42
                                    ))
            ])


bayes_model = GaussianNB()


# ============================================================
# 5. Modelle trainieren
# ============================================================

logistic_model.fit(X_train, y_train)
bayes_model.fit(X_train, y_train)


# ============================================================
# 6. Vorhersagen
# ============================================================

y_pred_logistic = logistic_model.predict(X_test)
y_pred_bayes = bayes_model.predict(X_test)


# ============================================================
# 7. Metriken berechnen
# ============================================================

def calculate_metrics(y_true, y_pred):

        return {
                        "Accuracy": accuracy_score(y_true, y_pred),
                                "Precision": precision_score(y_true, y_pred),
                                        "Recall": recall_score(y_true, y_pred),
                                                "F1": f1_score(y_true, y_pred)
                                                    }


        logistic_metrics = calculate_metrics(
                    y_test,
                        y_pred_logistic
                        )

        bayes_metrics = calculate_metrics(
                    y_test,
                        y_pred_bayes
                        )


        # ============================================================
        # 8. Ergebnisse ausgeben
        # ============================================================

        print("\nLOGISTIC REGRESSION")
        print("=" * 40)

        for metric, value in logistic_metrics.items():
                print(f"{metric}: {value:.4f}")
                print("\nGAUSSIAN NAIVE BAYES")
                print("=" * 40)
                for metric, value in bayes_metrics.items():
                        print(f"{metric}: {value:.4f}")
                        # ============================================================
                        # 9. Confusion Matrix - Logistic Regression
                        # ============================================================

                        ConfusionMatrixDisplay.from_predictions(
                                    y_test,
                                        y_pred_logistic,
                                            display_labels=["Benign", "Malignant"]
                                            )

                        plt.title("Confusion Matrix - Logistic Regression")
                        plt.tight_layout()
                        plt.show()
                        # ============================================================
                        # 10. Confusion Matrix - Gaussian Naive Bayes
                        # ============================================================
                        ConfusionMatrixDisplay.from_predictions(
                                    y_test,
                                        y_pred_bayes,
                                            display_labels=["Benign", "Malignant"]
                                            )
                        plt.title("Confusion Matrix - Gaussian Naive Bayes")
                        plt.tight_layout()
                        plt.show()
                        # ============================================================
                        # 11. ROC-Kurven vergleichen
                        # ============================================================
                        fig, ax = plt.subplots(figsize=(7, 6))
                        RocCurveDisplay.from_estimator(
                                    logistic_model,
                                        X_test,
                                            y_test,
                                                name="Logistic Regression",
                                                    ax=ax
                                                    )
                        RocCurveDisplay.from_estimator(
                                    bayes_model,
                                        X_test,
                                            y_test,
                                                name="Gaussian Naive Bayes",
                                                    ax=ax
                                                    )
                        # Zufallsklassifikator
                        ax.plot(
                                    [0, 1],
                                        [0, 1],
                                            linestyle="--",
                                                label="Random Classifier"
                                                )

                        ax.set_title("ROC Curve Comparison")
                        ax.legend()
                        plt.tight_layout()
                        plt.show()
                        # ============================================================
                        # 12. Vergleich der Metriken
                        # ============================================================

                        metric_names = [
                                    "Accuracy",
                                        "Precision",
                                            "Recall",
                                                "F1"
                                                ]
                        logistic_values = [
                                    logistic_metrics[metric]
                                        for metric in metric_names
                                        ]

                        bayes_values = [
                                    bayes_metrics[metric]
                                        for metric in metric_names
                                        ]


                        x = np.arange(len(metric_names))

                        width = 0.35

                        fig, ax = plt.subplots(figsize=(9, 5))

                        ax.bar(
                                    x - width / 2,
                                        logistic_values,
                                            width,
                                                label="Logistic Regression"
                                                )

                        ax.bar(
                                    x + width / 2,
                                        bayes_values,
                                            width,
                                                label="Gaussian Naive Bayes"
                                                )

                        ax.set_ylabel("Score")

                        ax.set_title(
                                    "Performance Comparison"
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
                        import pandas as pd
                        import numpy as np
                        import matplotlib.pyplot as plt

                        from sklearn.model_selection import train_test_split
                        from sklearn.pipeline import Pipeline
                        from sklearn.preprocessing import StandardScaler

                        from sklearn.linear_model import LogisticRegression
                        from sklearn.naive_bayes import GaussianNB

                        from sklearn.metrics import (
                                    accuracy_score,
                                        precision_score,
                                            recall_score,
                                                f1_score,
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
                        # ID und Diagnose sind keine Features
                        X = df.drop(columns=["id", "diagnosis"])

                        # 1 = malignant
                        # 0 = benign
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
                        # 4. Modelle definieren
                        # ============================================================
                        logistic_model = Pipeline([
                                ("scaler", StandardScaler()),
                                    ("model", LogisticRegression(
                                                max_iter=1000,
                                                        random_state=42
                                                            ))
                                    ])


                        bayes_model = GaussianNB()
                        # ============================================================
                        # 5. Modelle trainieren
                        # ============================================================

                        logistic_model.fit(X_train, y_train)
                        bayes_model.fit(X_train, y_train)
                        # ============================================================
                        # 6. Vorhersagen
                        # ============================================================
                        y_pred_logistic = logistic_model.predict(X_test)
                        y_pred_bayes = bayes_model.predict(X_test)
                        # ===========================================================
                        # 7. Metriken berechnen
                        # ============================================================

                        def calculate_metrics(y_true, y_pred):
                                return {
                                                "Accuracy": accuracy_score(y_true, y_pred),
                                                        "Precision": precision_score(y_true, y_pred),
                                                                "Recall": recall_score(y_true, y_pred),
                                                                        "F1": f1_score(y_true, y_pred)
                                                                            }
                                logistic_metrics = calculate_metrics(
                                            y_test,
                                                y_pred_logistic
                                                )

                                bayes_metrics = calculate_metrics(
                                            y_test,
                                                y_pred_bayes
                                                )
                                # ============================================================
                                # 8. Ergebnisse ausgeben
                                # ============================================================

                                print("\nLOGISTIC REGRESSION")
                                print("=" * 40)

                                for metric, value in logistic_metrics.items():
                                        print(f"{metric}: {value:.4f}")
                                        print("\nGAUSSIAN NAIVE BAYES")
                                        print("=" * 40)

                                        for metric, value in bayes_metrics.items():
                                                print(f"{metric}: {value:.4f}")
                                                # ============================================================
                                                # 9. Confusion Matrix - Logistic Regression
                                                # ============================================================
                                                ConfusionMatrixDisplay.from_predictions(
                                                            y_test,
                                                                y_pred_logistic,
                                                                    display_labels=["Benign", "Malignant"]
                                                                    )

                                                plt.title("Confusion Matrix - Logistic Regression")
                                                plt.tight_layout()
                                                plt.show()
                                                # ============================================================
                                                # 10. Confusion Matrix - Gaussian Naive Bayes
                                                # ============================================================
                                                ConfusionMatrixDisplay.from_predictions(
                                                            y_test,
                                                                y_pred_bayes,
                                                                    display_labels=["Benign", "Malignant"]
                                                                    )

                                                plt.title("Confusion Matrix - Gaussian Naive Bayes")
                                                plt.tight_layout()
                                                plt.show()
                                                # ============================================================
                                                # 11. ROC-Kurven vergleichen
                                                # ============================================================
                                                fig, ax = plt.subplots(figsize=(7, 6))


                                                RocCurveDisplay.from_estimator(
                                                            logistic_model,
                                                                X_test,
                                                                    y_test,
                                                                        name="Logistic Regression",
                                                                            ax=ax
                                                                            )


                                                RocCurveDisplay.from_estimator(
                                                            bayes_model,
                                                                X_test,
                                                                    y_test,
                                                                        name="Gaussian Naive Bayes",
                                                                            ax=ax
                                                                            )


                                                # Zufallsklassifikator
                                                ax.plot(
                                                            [0, 1],
                                                                [0, 1],
                                                                    linestyle="--",
                                                                        label="Random Classifier"
                                                                        )

                                                ax.set_title("ROC Curve Comparison")
                                                ax.legend()

                                                plt.tight_layout()
                                                plt.show()


                                                # ============================================================
                                                # 12. Vergleich der Metriken
                                                # ============================================================

                                                metric_names = [
                                                            "Accuracy",
                                                                "Precision",
                                                                    "Recall",
                                                                        "F1"
                                                                        ]


                                                logistic_values = [
                                                            logistic_metrics[metric]
                                                                for metric in metric_names
                                                                ]


                                                bayes_values = [
                                                            bayes_metrics[metric]
                                                                for metric in metric_names
                                                                ]


                                                x = np.arange(len(metric_names))

                                                width = 0.35


                                                fig, ax = plt.subplots(figsize=(9, 5))


                                                ax.bar(
                                                            x - width / 2,
                                                                logistic_values,
                                                                    width,
                                                                        label="Logistic Regression"
                                                                        )


                                                ax.bar(
                                                            x + width / 2,
                                                                bayes_values,
                                                                    width,
                                                                        label="Gaussian Naive Bayes"
                                                                        )


                                                ax.set_ylabel("Score")

                                                ax.set_title(
                                                            "Performance Comparison"
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
                                                plt.show()
