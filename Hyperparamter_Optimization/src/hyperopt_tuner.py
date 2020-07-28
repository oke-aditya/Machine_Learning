import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics

from sklearn import model_selection
from functools import partial
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, X, y):
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []
    for idx in kf.split(X=X, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = X[train_idx]
        ytrain = y[train_idx]

        xtest = X[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
    
    return -1.0 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    X = df.drop(["price_range"], axis=1).values
    y = df["price_range"].values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1,)

    param_space = {
        "max_depth" :  scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators" : scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion" : hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1), 
    }

    optimization_function = partial(optimize, X=X, y=y)
    
    trials = Trials()

    result = fmin(fn=optimization_function, space=param_space, trials=trials, max_evals=15, algo=tpe.suggest)

    print(result)

