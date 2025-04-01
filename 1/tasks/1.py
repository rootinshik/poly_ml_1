from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def get_accuracy_naive_gaussian(
    X: pd.DataFrame, y: pd.Series, train_size: float, random_state: int = 42
) -> tuple[float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    return (
        accuracy_score(y_train, model.predict(X_train)),
        accuracy_score(y_test, model.predict(X_test)),
    )


def plot_accuracy(data, name):
    plt.figure()
    plt.plot(data["perc"], data["train_acc"], label="Train")
    plt.plot(data["perc"], data["test_acc"], label="Test")
    plt.title(f"Данные: {name}")
    plt.xlabel("Процент обучающей выборки")
    plt.ylabel("(Точность) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./pics/1/{name}.png", dpi=250)
    plt.close()


def main():
    spam_df = pd.read_csv("./data/spam/spam.csv", index_col="Unnamed: 0")
    tic_tac_toe_df = pd.read_csv(
        "./data/tic_tac_toe.txt", names=list(range(9)) + ["target"]
    )
    tic_tac_toe_df.iloc[:, :-1] = tic_tac_toe_df.drop("target", axis=1) == "x"

    train_sizes = np.linspace(0.05, 0.95, 100).round(3)

    train_spam = partial(
        get_accuracy_naive_gaussian, X=spam_df.drop("type", axis=1), y=spam_df["type"]
    )

    train_tic_tac_toe = partial(
        get_accuracy_naive_gaussian,
        X=tic_tac_toe_df.drop("target", axis=1),
        y=tic_tac_toe_df["target"],
    )

    accuracy_scores = [
        (perc, train_spam(train_size=perc), train_tic_tac_toe(train_size=perc))
        for perc in train_sizes
    ]

    spam_df = pd.DataFrame(
        [(p, s[0], s[1]) for p, s, _ in accuracy_scores],
        columns=["perc", "train_acc", "test_acc"],
    )
    plot_accuracy(spam_df, "spam")

    tic_tac_toe_df = pd.DataFrame(
        [(p, t[0], t[1]) for p, _, t in accuracy_scores],
        columns=["perc", "train_acc", "test_acc"],
    )
    plot_accuracy(tic_tac_toe_df, "tic_tac_toe")


if __name__ == "__main__":
    main()
