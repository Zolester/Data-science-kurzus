"""
Impact of Social Media on Life - complete machine-learning homework.

Two related targets are analysed:
1. Academic_Performance_GPA: regression
2. Overall_Impact: three-class classification

The target column is never used as an input feature. GPA is also excluded from
the classification inputs because Overall_Impact is a multi-factor outcome
which may include academic performance; this avoids target leakage.

Run:
    python social_media_ml_homework.py

The script writes model comparison tables and plots to results/.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Social_media_impact_on_life.csv"
RESULTS = ROOT / "results"
RANDOM_STATE = 42


def make_preprocessor(features: pd.DataFrame, scale_numeric: bool = False):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def evaluate_regressors(X_train, X_test, y_train, y_test):
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Neural network": MLPRegressor(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500,
            random_state=RANDOM_STATE
        ),
    }
    rows = []
    predictions = {}
    for name, estimator in models.items():
        scale = name == "Neural network"
        model = Pipeline(
            [("preprocess", make_preprocessor(X_train, scale)), ("model", estimator)]
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predictions[name] = (model, prediction)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, prediction),
                "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                "R2": r2_score(y_test, prediction),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE"), predictions


def evaluate_classifiers(X_train, X_test, y_train, y_test):
    models = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Neural network": MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500,
            random_state=RANDOM_STATE
        ),
    }
    rows = []
    predictions = {}
    for name, estimator in models.items():
        scale = name in {"Logistic regression", "Neural network"}
        model = Pipeline(
            [("preprocess", make_preprocessor(X_train, scale)), ("model", estimator)]
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predictions[name] = (model, prediction)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Macro F1": f1_score(y_test, prediction, average="macro"),
            }
        )
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False), predictions


def save_eda(data: pd.DataFrame):
    RESULTS.mkdir(exist_ok=True)
    summary = data.describe(include="all").transpose()
    summary.to_csv(RESULTS / "data_summary.csv")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="Overall_Impact", order=data["Overall_Impact"].value_counts().index)
    plt.title("Overall impact class distribution")
    plt.tight_layout()
    plt.savefig(RESULTS / "impact_distribution.png", dpi=160)
    plt.close()

    numeric = data.select_dtypes(include="number")
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix of numeric variables")
    plt.tight_layout()
    plt.savefig(RESULTS / "correlation_matrix.png", dpi=160)
    plt.close()


def save_tree_visualization(model, feature_names):
    tree = model.named_steps["model"]
    transformed_names = model.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(22, 10))
    plot_tree(
        tree,
        feature_names=transformed_names,
        class_names=sorted(tree.classes_) if hasattr(tree, "classes_") else None,
        filled=True,
        max_depth=3,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "decision_tree.png", dpi=160)
    plt.close()


def save_permutation_importance(model, X_test, y_test, task):
    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring=scoring
    )
    table = pd.DataFrame(
        {"Feature": X_test.columns, "Importance": importance.importances_mean}
    ).sort_values("Importance", ascending=False)
    table.to_csv(RESULTS / f"{task}_feature_importance.csv", index=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=table.head(10), x="Importance", y="Feature", color="steelblue")
    plt.title(f"Top permutation features - {task}")
    plt.tight_layout()
    plt.savefig(RESULTS / f"{task}_feature_importance.png", dpi=160)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(data):,}; columns: {len(data.columns)}")
    print("\nMissing values:\n", data.isna().sum())
    print("\nImpact distribution:\n", data["Overall_Impact"].value_counts())
    save_eda(data)

    common = [
        "Age", "Gender", "Academic_Level", "Primary_Platform",
        "Daily_Usage_Hours", "Weekend_Extra_Hours", "Device_Type",
        "Sleep_Duration_Hours", "Sleep_Quality_Score", "Late_Night_Usage",
        "Social_Comparison_Frequency", "Perceived_Stress_Score",
        "Mental_Health_Index",
    ]

    regression_data = data.dropna(subset=["Academic_Performance_GPA"]).copy()
    X_reg = regression_data[common]
    y_reg = regression_data["Academic_Performance_GPA"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    regression_results, regression_models = evaluate_regressors(
        X_train_r, X_test_r, y_train_r, y_test_r
    )
    regression_results.to_csv(RESULTS / "regression_model_comparison.csv", index=False)
    print("\nGPA regression:\n", regression_results.to_string(index=False))

    X_cls = data[common]
    y_cls = data["Overall_Impact"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
    )
    classification_results, classification_models = evaluate_classifiers(
        X_train_c, X_test_c, y_train_c, y_test_c
    )
    classification_results.to_csv(RESULTS / "classification_model_comparison.csv", index=False)
    print("\nOverall impact classification:\n", classification_results.to_string(index=False))

    best_classifier_name = classification_results.iloc[0]["Model"]
    best_classifier, prediction = classification_models[best_classifier_name]
    print(f"\nBest classifier: {best_classifier_name}")
    print(classification_report(y_test_c, prediction))
    labels = sorted(y_cls.unique())
    pd.DataFrame(
        confusion_matrix(y_test_c, prediction, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(RESULTS / "best_classifier_confusion_matrix.csv")

    save_tree_visualization(classification_models["Decision tree"][0], common)
    save_permutation_importance(
        regression_models["Gradient boosting"][0], X_test_r, y_test_r, "regression"
    )
    save_permutation_importance(
        classification_models["Gradient boosting"][0], X_test_c, y_test_c, "classification"
    )

    print(f"\nResults written to: {RESULTS}")


if __name__ == "__main__":
    main()
