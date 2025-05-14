import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def generate_data(
    perc_pos: float,
    n_samples: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_state)

    n_pos = round(n_samples * perc_pos)
    mean_neg = [16, 11]
    std_neg = 3

    n_neg = round(n_samples * (1 - perc_pos))
    mean_pos = [17, 12]
    std_pos = 2

    X_neg = np.random.normal(loc=mean_neg, scale=std_neg, size=(n_neg, 2))
    X_pos = np.random.normal(loc=mean_pos, scale=std_pos, size=(n_pos, 2))

    X = np.vstack([X_neg, X_pos])
    y = np.array([-1] * n_neg + [1] * n_pos)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y
    return df


def main():
    df = generate_data(perc_pos=0.6)
    print("Проверка распределения:")
    print(df.groupby("target").agg(["mean", "std", "count"]))

    for class_ in df["target"].unique():
        class_df = df.query(f"target == {class_}")
        plt.scatter(
            class_df["x1"],
            class_df["x2"],
            label=f"Класс {class_}",
        )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.savefig("./pics/2/data_distribution.png", dpi=250)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1),
        df["target"],
        test_size=0.25,
        random_state=42,
        stratify=df["target"],
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"{accuracy=}")

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    cm_plot = ConfusionMatrixDisplay(cm, display_labels=df["target"].unique())
    cm_plot.plot()

    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")

    plt.savefig("./pics/2/confusion_matrix.png", dpi=250)
    plt.close()

    roc_curve_plot = RocCurveDisplay.from_estimator(model, X_test, y_test, pos_label=-1)

    plt.title("ROC кривая")
    plt.ylabel("TPR (Recall)")
    plt.xlabel("FPR")

    plt.savefig("./pics/2/roc_curve.png", dpi=250)

    pr_curve_plot = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, pos_label=1
    )

    plt.title("PR кривая")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig("./pics/2/pr_curve.png", dpi=250)
    plt.close()


if __name__ == "__main__":
    main()
