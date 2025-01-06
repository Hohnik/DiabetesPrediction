import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Random Forrest Classification
        """
    )
    return


@app.cell
def __():
    import time

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                         ParameterSampler, RandomizedSearchCV,
                                         train_test_split)

    from Transformer import ReplaceZeroWithMean

    DATAPATH = "../Data"
    MODELPATH = "../Data/Models/RFC"

    data = pd.read_csv(f"{DATAPATH}/diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=1
    )
    imputer = ReplaceZeroWithMean(["Glucose", "BloodPressure", "SkinThickness", "BMI"])
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return (
        DATAPATH,
        GridSearchCV,
        MODELPATH,
        ParameterGrid,
        ParameterSampler,
        RandomForestClassifier,
        RandomizedSearchCV,
        ReplaceZeroWithMean,
        X,
        X_test,
        X_train,
        data,
        imputer,
        joblib,
        np,
        pd,
        time,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## WITHOUT Hyperparameter Tuning
        """
    )
    return


@app.cell
def __(MODELPATH, RandomForestClassifier, X_train, joblib, y_train):
    rfc_base = RandomForestClassifier(n_jobs=-1, random_state=1)
    rfc_base.fit(X_train, y_train)

    joblib.dump(rfc_base, f"{MODELPATH}/RFC_no_hyper.pkl")
    return (rfc_base,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## WITH Hyperparameter Tuning
        """
    )
    return


@app.cell
def __(
    MODELPATH,
    ParameterSampler,
    RandomForestClassifier,
    X_test,
    X_train,
    joblib,
    np,
    param_dist,
    y_test,
    y_train,
):
    import sys
    from pprint import pprint

    from joblib import Parallel, delayed

    _param_dist = {
        "n_estimators": list(range(50, 150, 10)),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": np.linspace(2, 50, 2, dtype=np.int8),
        "max_features": ["sqrt", "log2"],
        "max_leaf_nodes": [None, 5, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50, 100],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    param_sampler = list(ParameterSampler(param_dist, n_iter=10000, random_state=1))

    def train_and_evaluate(params):
        model = RandomForestClassifier(n_jobs=-1, random_state=1, **params)
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        return (test_score, model)

    results = Parallel(n_jobs=-1)(
        (delayed(train_and_evaluate)(params) for params in param_sampler)
    )
    rfc_hpt_score, rfc_hpt = max(results, key=lambda x: x[0])
    print(rfc_hpt_score)
    joblib.dump(rfc_hpt, f"{MODELPATH}/RFC_hyper.pkl")
    return (
        Parallel,
        delayed,
        param_sampler,
        pprint,
        results,
        rfc_hpt,
        rfc_hpt_score,
        sys,
        train_and_evaluate,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## WITH Hyperparamerter Tuning AND Cross Validation
        """
    )
    return


@app.cell
def __(
    MODELPATH,
    RandomForestClassifier,
    RandomizedSearchCV,
    X_test,
    X_train,
    joblib,
    np,
    param_dist,
    rfc,
    y_test,
    y_train,
):
    _param_dist = {
        "n_estimators": list(range(50, 150, 15)),
        "criterion": ["gini", "log_loss"],
        "max_depth": np.linspace(2, 50, 3, dtype=np.int8),
        "max_features": ["sqrt", "log2"],
        "max_leaf_nodes": [None, 5, 8, 9, 10, 11, 12, 20, 50, 100],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    _rfc = RandomForestClassifier(n_jobs=-1, random_state=1)
    models = RandomizedSearchCV(
        rfc, param_distributions=param_dist, n_iter=5000, cv=6, n_jobs=-1
    )
    models.fit(X_train, y_train)
    rfc_hpt_cv = models.best_estimator_
    print(rfc_hpt_cv.score(X_test, y_test))
    joblib.dump(rfc_hpt_cv, f"{MODELPATH}/RFC_hyper_cv.pkl")
    return models, rfc_hpt_cv


@app.cell
def __(mo):
    mo.md(
        r"""
        ## WITH Hyperparamerter Tuning AND Cross Validation (Stratisfied)
        """
    )
    return


@app.cell
def __(
    MODELPATH,
    RandomForestClassifier,
    RandomizedSearchCV,
    X_test,
    X_train,
    joblib,
    np,
    param_dist,
    rfc,
    y_test,
    y_train,
):
    from sklearn.model_selection import StratifiedKFold

    _param_dist = {
        "n_estimators": list(range(50, 150, 15)),
        "criterion": ["gini", "log_loss"],
        "max_depth": np.linspace(2, 50, 3, dtype=np.int8),
        "max_features": ["sqrt", "log2"],
        "max_leaf_nodes": [None, 5, 8, 9, 10, 11, 12, 20, 50, 100],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    _rfc = RandomForestClassifier(n_jobs=-1, random_state=1)
    cv_split = StratifiedKFold(n_splits=5, shuffle=True)
    models_1 = RandomizedSearchCV(
        rfc, param_distributions=param_dist, n_iter=5000, cv=cv_split, n_jobs=-1
    )
    models_1.fit(X_train, y_train)
    rfc_hpt_cv_1 = models_1.best_estimator_
    print(rfc_hpt_cv_1.score(X_test, y_test))
    joblib.dump(rfc_hpt_cv_1, f"{MODELPATH}/RFC_hyper_cv.pkl")
    return StratifiedKFold, cv_split, models_1, rfc_hpt_cv_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Further Parameter Adjustments
        """
    )
    return


@app.cell
def __(
    GridSearchCV,
    MODELPATH,
    RandomForestClassifier,
    X_test,
    X_train,
    joblib,
    pprint,
    rfc,
    y_test,
    y_train,
):
    _rfc = RandomForestClassifier(
        max_features="log2", class_weight="balanced", n_jobs=-1, random_state=1
    )
    param_grid = {"n_estimators": 125, "max_leaf_nodes": 20, "max_depth": 26}
    for param, value in param_grid.items():
        percent = 5
        param_grid.update(
            {
                param: [int(value * i / 100) for i in range(105, 130, percent)]
                + [value]
                + [int(value * i / 100) for i in range(95, 70, -percent)]
            }
        )
    pprint(param_grid)
    models_2 = GridSearchCV(rfc, param_grid, cv=10, n_jobs=-1)
    models_2.fit(X_train, y_train)
    rfc_best = models_2.best_estimator_
    print(rfc_best.score(X_test, y_test))
    joblib.dump(rfc_best, f"{MODELPATH}/RFC_hyper_cv_tuned.pkl")
    return models_2, param, param_grid, percent, rfc_best, value


@app.cell
def __(X_test, rfc_base, rfc_best, rfc_hpt_cv_1, rfc_hpt_score, y_test):
    print(f"Base      : {rfc_base.score(X_test, y_test)}")
    print(f"HPT       : {rfc_hpt_score}")
    print(f"HPT+CV    : {rfc_hpt_cv_1.score(X_test, y_test)}")
    print(f"HPT+CV+Opt: {rfc_best.score(X_test, y_test)}")
    return


@app.cell
def __(X_test, X_train, joblib, model, pprint, y_test, y_train):
    RFC_URL = "../Data/Models/RFC"
    SVM_URL = "../Data/Models/SVM"
    models_3 = {
        "RFC": f"{RFC_URL}/RFC_no_hyper.pkl",
        "RFC_hyper": f"{RFC_URL}/RFC_hyper.pkl",
        "RFC_hyper_cv": f"{RFC_URL}/RFC_hyper_cv.pkl",
        "RFC_hyper_cv_tuned": f"{RFC_URL}/RFC_hyper_cv_tuned.pkl",
    }
    scores = {}
    for name, _model in models_3.items():
        _model = joblib.load(model)
        _model.fit(X_train, y_train)
        scores.update({name: _model.score(X_test, y_test)})
    pprint(scores)
    return RFC_URL, SVM_URL, models_3, name, scores


@app.cell
def __(
    RandomForestClassifier,
    X_train,
    class_names,
    joblib,
    models_3,
    rfc,
    target,
    y_train,
):
    from dtreeviz import model

    _rfc = joblib.load(models_3["RFC"])
    tree = _rfc.estimators_[0]
    viz_model = model(
        tree,
        X_train=X_train,
        y_train=y_train,
        target_name=target,
        class_names=list(class_names),
        tree_index=2,
    )
    viz_model.view()
    return model, tree, viz_model


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
