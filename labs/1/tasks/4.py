import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(train_path: str, test_path: str) -> tuple:
    train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")

    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


def plot_decision_boundary(
    model, X: np.ndarray, y: np.ndarray, title: str, path_to_save: str = None
):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        cmap=plt.cm.Paired,
        alpha=0.6,
        ax=ax,
        plot_method="contourf",
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=40,
        edgecolors="k",
        label="Данные",
    )

    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=1.5,
        label="Опорные векторы",
    )

    if model.support_vectors_ is not None:
        plt.title(f"{title}\nОпорные векторы: {len(model.support_vectors_)}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{path_to_save}.png", dpi=250)
    plt.close()


def plot_confusion_matrix(
    model, X: np.ndarray, y: np.ndarray, dataset: str, path_to_save: str = None
):
    cm = confusion_matrix(y, model.predict(X))
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Матрица ошибок ({dataset})")
    plt.savefig(f"{path_to_save}.png", dpi=250)
    plt.close()


def a():
    X_train, y_train, X_test, y_test = load_data(
        "./data/svmdata/svmdata_a.txt", "./data/svmdata/svmdata_a_test.txt"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train_scaled, y_train)

    plot_decision_boundary(
        model,
        X_train_scaled,
        y_train,
        "SVM с линейным ядром",
        path_to_save="./pics/4/a/Linear_SVM.png",
    )
    plot_confusion_matrix(
        model,
        X_train_scaled,
        y_train,
        "Обучение",
        path_to_save="./pics/4/a/Confusion_Matrix_Train.png",
    )
    plot_confusion_matrix(
        model,
        X_test_scaled,
        y_test,
        "Тест",
        path_to_save="./pics/4/a/Confusion_Matrix_Test.png",
    )


def b():
    X_train, y_train, X_test, y_test = load_data(
        "./data/svmdata/svmdata_b.txt", "./data/svmdata/svmdata_b_test.txt"
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    c_values = np.logspace(-3, 5, 100)
    train_errors = []
    for c in c_values:
        model = SVC(kernel="linear", C=c)
        model.fit(X_train_scaled, y_train)
        train_errors.append(1 - model.score(X_train_scaled, y_train))

    best_train_c = c_values[np.argmin(train_errors)]
    print(f"{best_train_c=}")

    test_errors = [
        1
        - SVC(kernel="linear", C=c)
        .fit(X_train_scaled, y_train)
        .score(X_test_scaled, y_test)
        for c in c_values
    ]
    best_test_c = c_values[np.argmin(test_errors)]
    print(f"{best_test_c=}")

    gs = GridSearchCV(SVC(kernel="linear"), {"C": c_values}, cv=5)
    gs.fit(X_train_scaled, y_train)

    plt.semilogx(c_values, train_errors, label="Ошибка на обучении")
    plt.semilogx(c_values, test_errors, label="Ошибка на тесте")
    plt.scatter(gs.best_params_["C"], gs.best_score_, c="red", label="Лучшее CV")
    plt.legend()
    plt.title("Ошибка в зависимости от $C$")
    plt.savefig("./pics/4/b/Error_vs_C.png", dpi=250)
    plt.close()

    print(f"Оптимальное C: {gs.best_params_['C']}")


def c():
    X_train, y_train, X_test, y_test = load_data(
        "./data/svmdata/svmdata_c.txt", "./data/svmdata/svmdata_c_test.txt"
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernels = [
        ("linear", {}),
        ("poly", {"degree": 1}),
        ("poly", {"degree": 3}),
        ("poly", {"degree": 5}),
        ("sigmoid", {}),
        ("rbf", {}),
    ]

    for kernel in kernels:
        model = SVC(kernel=kernel[0], **kernel[1])
        model.fit(X_train_scaled, y_train)
        plot_decision_boundary(
            model,
            X_test_scaled,
            y_test,
            f"SVM с ядром {kernel[0]} {kernel[1]}",
            path_to_save=f"./pics/4/c/SVM_{kernel[0]}_{kernel[1]}.png",
        )


def d():
    X_train, y_train, X_test, y_test = load_data(
        "./data/svmdata/svmdata_d.txt", "./data/svmdata/svmdata_d_test.txt"
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernels = [
        ("poly", {"degree": 1}),
        ("poly", {"degree": 3}),
        ("poly", {"degree": 5}),
        ("sigmoid", {}),
        ("rbf", {}),
    ]

    for kernel in kernels:
        model = SVC(kernel=kernel[0], **kernel[1])
        model.fit(X_train_scaled, y_train)
        plot_decision_boundary(
            model,
            X_test_scaled,
            y_test,
            f"SVM с ядром {kernel[0]} {kernel[1]}",
            path_to_save=f"./pics/4/d/SVM_{kernel[0]}_{kernel[1]}.png",
        )


def e():
    X_train, y_train, X_test, y_test = load_data(
        "./data/svmdata/svmdata_e.txt", "./data/svmdata/svmdata_e_test.txt"
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gammas = [0.1, 1, 10, 100]
    for gamma in gammas:
        model = SVC(kernel="rbf", gamma=gamma)
        model.fit(X_train_scaled, y_train)
        plot_decision_boundary(
            model,
            X_test_scaled,
            y_test,
            f"RBF SVM с gamma={gamma}",
            path_to_save=f"./pics/4/e/RBF_SVM_gamma_{gamma}.png",
        )


def main():
    tasks = [a, b, c, d, e]
    for task in tasks:
        print(task.__name__)
        task()
        print()


if __name__ == "__main__":
    main()
