import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    fbeta_score,
    make_scorer,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


ftwo_scorer = make_scorer(fbeta_score, beta=0.75)


def prepare_data(train_path: str, test_path: str, target_col: str) -> tuple:
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def plot_feature_importance(model, feature_names, model_name):
    plt.figure(figsize=(12, 8))

    if isinstance(model, LogisticRegression):
        importance = np.abs(model.coef_[0])
    else:
        importance = model.get_feature_importance()

    indices = np.argsort(importance)[::-1]

    plt.title(f"Важность признаков ({model_name})")
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Важность")
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"./pics/6/{model_name}_feature_importance.png", dpi=250)
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1)
    ax1.set_title(f"ROC Curve ({model_name})")
    ax1.grid(True)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2)
    ax2.set_title(f"PR Curve ({model_name})")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"./pics/6/{model_name}_roc_pr_curves.png", dpi=250)
    plt.close()

    return {
        "f_beta": fbeta_score(y_test, y_pred, beta=0.75),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def logistic_regression_model(X_train, X_test, y_train, feature_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {"C": np.logspace(-3, 3, 7), "class_weight": [None, "balanced"]}

    lr = GridSearchCV(
        LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000),
        param_grid,
        verbose=1,
        scoring=ftwo_scorer,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    lr.fit(X_train_scaled, y_train)

    plot_feature_importance(lr.best_estimator_, feature_names, "Logistic Regression")

    return lr.best_estimator_, X_test_scaled


def catboost_model(X_train, X_test, y_train, feature_names):
    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.3],
        "iterations": [50, 100, 250],
    }

    cb = GridSearchCV(
        CatBoostClassifier(silent=True),
        param_grid,
        scoring=ftwo_scorer,
        verbose=2,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    cb.fit(X_train, y_train)

    plot_feature_importance(cb.best_estimator_, feature_names, "CatBoost")

    return cb.best_estimator_, X_test


def compare_models(train_path: str, test_path: str, target_col: str):
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path, target_col)
    feature_names = X_train.columns.tolist()

    lr_model, lr_X_test = logistic_regression_model(
        X_train, X_test, y_train, feature_names
    )
    cb_model, cb_X_test = catboost_model(X_train, X_test, y_train, feature_names)

    lr_metrics = evaluate_model(lr_model, lr_X_test, y_test, "Logistic Regression")
    cb_metrics = evaluate_model(cb_model, cb_X_test, y_test, "CatBoost")

    comparison_df = pd.DataFrame(
        {
            "Logistic Regression": [
                lr_model.get_params(),
                lr_metrics["f_beta"],
                lr_metrics["accuracy"],
            ],
            "CatBoost": [
                cb_model.get_params(),
                cb_metrics["f_beta"],
                cb_metrics["accuracy"],
            ],
        },
        index=["Параметры", "F0.75 Score", "Accuracy"],
    )

    print("\nСравнение моделей на тестовых данных:")
    return comparison_df


if __name__ == "__main__":
    result = compare_models(
        train_path="./data/bank_scoring/bank_scoring_train.csv",
        test_path="./data/bank_scoring/bank_scoring_test.csv",
        target_col="SeriousDlqin2yrs",
    )
    print(result)
    result.to_csv("./results/results.csv")
