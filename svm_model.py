import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import optuna
from joblib import Parallel, delayed
from pprint import pprint


# optunaの最適化結果を格納するためのDB作成
db_name = 'optuna.db'
conn = sqlite3.connect(db_name)


def main():
    data = pd.read_csv("blink_data/balanced_preproc_all.csv")
    X = data.drop(["frame", "blink"], axis=1)
    y = data["blink"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    scalar = StandardScaler()
    scaledX_train = scalar.fit_transform(X_train)
    scaledX_test = scalar.transform(X_test)
    arg_for_optuna = [scaledX_train, y_train, scaledX_test, y_test]

    study = optuna.create_study(
        study_name="svm_study",
        direction="maximize",
        storage="sqlite:///./optuna.db",
        load_if_exists=True
    )
    if not study.best_params:
        Parallel(n_jobs=4)([delayed(pass_arg_to_run(pass_data_to_objective(*arg_for_optuna), n_trials=500))() for _ in range(4)])
        # 並列化しない場合
        # study.optimize(pass_data_to_objective(*arg_for_optuna), n_trials=500)

    c = study.best_params["C"]
    kernel = study.best_params["kernel"]
    gamma = study.best_params["gamma"]

    svm = SVC(C=c, kernel=kernel, gamma=gamma, random_state=0)
    svm.fit(scaledX_train, y_train)

    predict = svm.predict(scaledX_test)
    acc = metrics.roc_auc_score(y_test, predict)

    pprint(acc)
    conn.close()


def pass_arg_to_run(objective, n_trials=500):
    """
    optunaの最適化を並列化するための関数
    参考: https://ohke.hateblo.jp/entry/2020/07/04/230000
    """
    def run():
        study = optuna.load_study(study_name="svm_study", storage="sqlite:///./optuna.db")
        study.optimize(objective, n_trials=n_trials)
        return os.getpid()

    return run


def pass_data_to_objective(X_train, y_train, X_test, y_test):
    """
    訓練データと検証データを渡すための高階関数
    参考: https://tech.515hikaru.net/post/2019-06-26-optuna-have-arg/
    """
    def objective(trial):
        # ハイパーパラメータの設定
        c = trial.suggest_loguniform("C", 1e-3, 1e3)
        kernel = trial.suggest_categorical("kernel", ['linear', 'rbf', 'sigmoid'])
        gamma = trial.suggest_loguniform("gamma", 0.001, 1000)

        svm = SVC(C=c, kernel=kernel, gamma=gamma, random_state=0)
        svm.fit(X_train, y_train)

        # モデルの評価
        y_predict = svm.predict(X_test)
        auc = metrics.roc_auc_score(y_test, y_predict)

        return auc

    return objective


if __name__ == "__main__":
    main()
