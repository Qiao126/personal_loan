from mpl_toolkits.mplot3d import Axes3D
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import argparse
from utils import feature_discretizaiton
from plot import plot_learning_curve, plot_feat_distribution, plot_PCA_analysis, plot_curve, plot_feat, plot_feature_importance

ratio = 0.2


def preprocessing(data_file, feature_discret=False, plot_feat_dist=False, plot_PCA=False, normalize=False):
    data = pd.read_csv(data_file)
    cols = [
        'Age',
        'Experience',
        'Income',
        'ZIP Code',
        'Family',
        'CCAvg',
        'Education',
        'Mortgage',
        'Personal Loan',
        'Securities Account',
        'CD Account',
        'Online',
        'CreditCard']

    data = data[cols]

    print("Data shape", data.shape)
    print(data.describe(include='all'))

    y = data['Personal Loan']
    print(data.head(10))

    data = data.drop(['Personal Loan'], 1)
    X = data
    le = LabelEncoder()

    if plot_PCA:
        plot_PCA_analysis(X, y, 2)

    if plot_feat_dist:
        plot_feat_distribution(X, X.columns)

    if feature_discret:
        bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100]
        X = feature_discretizaiton(X, 'Age', bins)

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return data, X, y


def train_gbdt_classifier(X, y):
    print('start training gbdt classifier')
    clf = HistGradientBoostingClassifier()
    clf.fit(X, y)
    return clf


def train_random_forest_classifier(class_weight, X, y):
    print('start training random forest classifier')
    clf = RandomForestClassifier(
        n_estimators=30, max_depth=10, class_weight=class_weight)
    clf.fit(X, y)
    return clf


def train_logit_classifier(class_weight, X, y):
    print('strat training logistic classifier')
    clf = SGDClassifier(loss='log', class_weight=class_weight)
    clf.fit(X, y)
    return clf


def train_MLP_classifier(class_weight):
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), verbose=0, random_state=0, class_weight=class_weight)


def main():
    parser = argparse.ArgumentParser(description='Classification task')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--plot-pca', action='store_true')
    parser.add_argument('--plot-feature-dist', action='store_true')
    parser.add_argument('--plot-curve', action='store_true')
    parser.add_argument('--classifier')
    parser.add_argument('datafile')

    args = parser.parse_args()

    data, X, y = preprocessing(
        args.datafile,
        plot_feat_dist=args.plot_feature_dist,
        plot_PCA=args.plot_pca,
        normalize=args.normalize)

    train_num = int(data.shape[0] * (1-2*ratio))
    dev_num = int(data.shape[0] * (1-ratio))
    X_train, X_dev, X_test = X[:train_num], X[train_num:dev_num], X[dev_num:]
    y_train, y_dev, y_test = y[:train_num], y[train_num:dev_num], y[dev_num:]

    print("DataSet summary:")
    print("train X: ", X_train.shape, "y: ", y_train.shape)
    print("dev X: ", X_dev.shape, "y: ", y_dev.shape)
    print("test X:", X_test.shape, "y: ", y_test.shape)

    positive_weight = 1 - sum(y_train) / y_train.count()
    class_weight = {1: positive_weight, 0: 1-positive_weight}
    print({1: positive_weight, 0: 1-positive_weight})

    if args.classifier == 'gbdt':
        clf = train_gbdt_classifier(X_train, y_train)
    elif args.classifier == 'rf':
        clf = train_random_forest_classifier(class_weight, X_train, y_train)
    else:
        clf = train_logit_classifier(class_weight, X_train, y_train)

    title = "Learning Curve"
    plot_learning_curve(clf, title, X, y, cv=None, n_jobs=1)
    plt.show()

    plot_feature_importance(clf, X_train, y_train, data.columns)

    y_train_pred = clf.predict(X_train)
    y_train_prop = clf.predict_proba(X_train)
    y_dev_pred = clf.predict(X_dev)
    y_dev_prop = clf.predict_proba(X_dev)
    y_test_pred = clf.predict(X_test)
    y_test_prop = clf.predict_proba(X_test)
    y_train_score = y_train_prop[:, 1]
    y_dev_score = y_dev_prop[:, 1]
    y_test_score = y_test_prop[:, 1]

    print('Training data report', classification_report(y_train, y_train_pred))
    print('Dev data report', classification_report(y_dev, y_dev_pred))
    print('Test data report', classification_report(y_test, y_test_pred))
    opthd = plot_curve(y_dev, y_dev_score, min_p=None, min_r=None)

if __name__ == '__main__':
    main()
