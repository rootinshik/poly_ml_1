import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def evaluate_knn(
    n_neighbors: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = "minkowski",
) -> tuple[float]:
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    scaller = StandardScaler()
    model.fit(X=scaller.fit_transform(X_train), y=y_train)

    train_pred = model.predict(scaller.transform(X_train))
    test_pred = model.predict(scaller.transform(X_test))

    return train_pred, test_pred


def plot_metric(
    df: pd.DataFrame, key: str, first_col: str, second_col: str, ylabel: str, title: str
) -> None:
    plt.plot(df[key], df[first_col], label="Обучающая выборка")

    plt.plot(df[key], df[second_col], label="Тестовая выборка")

    plt.xticks(df[key][::2], rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(key)
    plt.legend()

    plt.grid()
    plt.savefig(f"./pics/3/{df.name}_{key}_{first_col}_{second_col}.png", dpi=250)

    plt.close()


def main():
    glass_df = pd.read_csv("./data/glass.csv").drop("Id", axis=1)
    glass_df = glass_df.rename(columns=str.lower)
    print(glass_df)

    X_train, X_test, y_train, y_test = train_test_split(
        glass_df.drop("type", axis=1),
        glass_df["type"],
        test_size=0.15,
        stratify=glass_df["type"],
        random_state=42,
    )

    neighbors_range = np.arange(1, 50, 1)
    train_accuracy, test_accuracy = [], []

    for k in neighbors_range:
        scores = evaluate_knn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_neighbors=k,
        )
        train_acc = accuracy_score(y_pred=scores[0], y_true=y_train)
        test_acc = accuracy_score(y_pred=scores[1], y_true=y_test)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    accuracy_df = pd.DataFrame(
        zip(neighbors_range, train_accuracy, test_accuracy),
        columns=["n", "train_acc", "test_acc"],
    )
    accuracy_df.name = "accuracy"

    accuracy_df["train_error"] = 1 - accuracy_df["train_acc"]
    accuracy_df["test_error"] = 1 - accuracy_df["test_acc"]

    plot_metric(
        accuracy_df,
        "n",
        "train_acc",
        "test_acc",
        "Точность",
        "Зависимость Точности (Accuracy) от количества соседей",
    )

    plot_metric(
        accuracy_df,
        "n",
        "train_error",
        "test_error",
        "Ошибка",
        "Зависимость ошибки (1 - Accuracy) от количества соседей",
    )

    best_n = int(accuracy_df.loc[accuracy_df["test_acc"].idxmax()]["n"])
    print(best_n)

    metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    train_accuracy, test_accuracy = [], []

    for metric in metrics:
        scores = evaluate_knn(
            n_neighbors=best_n,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metric=metric,
        )
        train_acc = accuracy_score(y_pred=scores[0], y_true=y_train)
        test_acc = accuracy_score(y_pred=scores[1], y_true=y_test)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    accuracy_df = pd.DataFrame(
        zip(metrics, train_accuracy, test_accuracy),
        columns=["metric", "train_acc", "test_acc"],
    )
    accuracy_df.name = "accuracy"

    accuracy_df["train_error"] = 1 - accuracy_df["train_acc"]
    accuracy_df["test_error"] = 1 - accuracy_df["test_acc"]

    print(accuracy_df)

    best_metric = accuracy_df.loc[accuracy_df["test_acc"].idxmax()]["metric"]
    print(best_metric)

    model = KNeighborsClassifier(n_neighbors=best_n, metric=best_metric)

    scaller = StandardScaler()
    model.fit(X=scaller.fit_transform(X_train), y=y_train)

    sample = {
        "RI": 1.516,
        "Na": 11.7,
        "Mg": 1.01,
        "Al": 1.19,
        "Si": 72.59,
        "K": 0.43,
        "Ca": 11.44,
        "Ba": 0.02,
        "Fe": 0.1,
    }

    sample_df = pd.DataFrame(sample, index=[0])
    sample_df = sample_df.rename(columns=str.lower)
    print(sample_df)

    sample_class = model.predict(scaller.transform(sample_df))
    print(sample_class)


if __name__ == "__main__":
    main()
