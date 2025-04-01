import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def load_glass_data():
    df = pd.read_csv("./data/glass.csv")
    df = df.rename(columns=str.lower).drop("id", axis=1)
    X = df.drop("type", axis=1)
    y = df["type"]
    return X, y


def visualize_tree(tree_model, feature_names, class_names, filename):
    dot_data = export_graphviz(
        tree_model,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename, format="png", cleanup=True)
    return graph


def analyze_parameters(X, y):
    """Анализ влияния параметров дерева с отдельными графиками для каждого параметра"""
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
    }

    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs.fit(X, y)
    results = pd.DataFrame(gs.cv_results_)

    plt.figure(figsize=(10, 6))
    results_agg_df = results.groupby("param_criterion")["mean_test_score"].mean()
    print(results_agg_df)
    results_agg_df.plot.bar()
    plt.title("Зависимость точности от критерия расщепления")
    plt.xlabel("Критерий")
    plt.grid(True)
    plt.ylabel("Средняя точность")
    plt.ylim(0.4, 1.0)
    plt.savefig("./pics/5/glass_criterion_analysis.png", dpi=250, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    results["param_max_depth"] = results["param_max_depth"].fillna("None").astype(str)
    results_agg_df = results.groupby("param_max_depth")["mean_test_score"].mean()
    print(results_agg_df)
    results_agg_df.plot(marker="o", linestyle="--")
    plt.title("Зависимость точности от максимальной глубины")
    plt.xlabel("Максимальная глубина")
    plt.ylabel("Средняя точность")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.savefig("./pics/5/glass_depth_analysis.png", dpi=250, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    results_agg_df = results.groupby("param_min_samples_split")[
        "mean_test_score"
    ].mean()
    print(results_agg_df)
    results_agg_df.plot(kind="bar", color="orange")
    plt.title("Зависимость точности от min_samples_split")
    plt.xlabel("Минимальное число образцов для разделения")
    plt.ylabel("Средняя точность")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.savefig(
        "./pics/5/glass_samples_split_analysis.png", dpi=250, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    for criterion in ["gini", "entropy"]:
        subset = results[results["param_criterion"] == criterion]
        subset.groupby("param_max_depth")["mean_test_score"].mean().plot(
            label=f"Criterion: {criterion}", marker="o"
        )
    plt.title("Совместное влияние глубины и критерия")
    plt.xlabel("Максимальная глубина")
    plt.ylabel("Средняя точность")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "./pics/5/glass_depth_criterion_combined.png", dpi=250, bbox_inches="tight"
    )
    plt.close()

    print("Анализ параметров завершен. Графики сохранены в ./pics/5/")
    return gs.best_estimator_


def a():
    X, y = load_glass_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    base_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    print(f"Базовая точность: {base_tree.score(X_test, y_test):.2f}")

    visualize_tree(base_tree, X.columns, y.unique(), "./pics/5/glass_tree")

    best_tree = analyze_parameters(X, y)
    print(f"Лучшие параметры: {best_tree.get_params()}")

    print(f"Глубина дерева: {best_tree.get_depth()}")
    print(f"Количество листьев: {best_tree.get_n_leaves()}")


def load_spam_data():
    df = pd.read_csv("./data/spam/spam7.csv")
    X = df.drop("yesno", axis=1)
    y = df["yesno"]
    return X, y


def optimize_spam_tree():
    X, y = load_spam_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_grid = {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    gs = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring="accuracy", cv=5)
    gs.fit(X_train, y_train)

    best_tree = gs.best_estimator_
    print(f"Точность оптимального дерева: {best_tree.score(X_test, y_test):.2f}")

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, best_tree.feature_importances_)
    plt.title("Важность признаков")
    plt.savefig("./pics/5/spam_feature_importance.png", dpi=250)
    plt.grid(True)
    plt.close()

    return best_tree, X_test, y_test, y


def b():
    best_tree, X_test, y_test, y = optimize_spam_tree()

    visualize_tree(best_tree, X_test.columns, y.unique(), "./pics/5/spam_tree")

    cm = confusion_matrix(y_test, best_tree.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=y.unique())
    disp.plot()
    plt.title("Матрица ошибок")
    plt.savefig("./pics/5/spam_confusion_matrix.png", dpi=250)
    plt.close()


def main():
    tasks = [a, b]
    for task in tasks:
        print(task.__name__)
        task()
        print()


if __name__ == "__main__":
    main()
