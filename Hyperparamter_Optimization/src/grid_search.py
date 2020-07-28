import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics

from sklearn import model_selection
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    X = df.drop(["price_range"], axis=1).values
    y = df["price_range"].values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1,)

    clf = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("rf", rf),])

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 5, 7],
    }

    param_distributions = {
        "pca__n_components": np.arange(5, 10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1, 20),
        "rf__criterion": ["gini", "entropy"],
    }

    # model = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, scoring="accuracy", verbose=10, n_jobs=-1, cv=5)
    model = model_selection.RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=-1,
        cv=5,
    )
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
