from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import time
import numpy as np
import pandas as pd
import os

def k_fold_training(classifier, x_all, y_all, x_valid, y_valid, folds, nb_classes):

    # Metrics for performance evaluation
    acc_per_fold = []
    epoch_per_fold = []
    loss_per_fold = []
    rec_per_fold = []
    prec_per_fold = []
    f1_per_fold = []
    duration_per_fold = []
    prediction_per_fold = []

    # Metrics for binary classification
    if nb_classes == 2:
        df_roc_curve = pd.DataFrame(columns = range(1, folds))
        # Array to compute the mean false positive rate (100 points)
        mean_fpr = np.linspace(0, 1, 100)
        auroc_per_fold = []

    sdk = StratifiedKFold(n_splits = folds, random_state = 42, shuffle = True)

    y_all_raw = np.argmax(y_all, axis = 1)

    for n_fold, (train, test) in enumerate(sdk.split(x_all, y_all_raw)):

        start_time_fold = time.time()

        # Training the classifier
        classifier.fit(x_all[train], y_all[train, :], x_valid, y_valid, fold = n_fold + 1)

        duration_fold = time.time() - start_time_fold
        duration_per_fold.append(duration_fold)

        # Predicting using the trained classifier
        t_0_pred = time.time()
        y_pred = classifier.predict(x_all[test], y_all[test, :], x_all[train], y_all[train, :], return_df_metrics = False, nb_classes = nb_classes)
        t_f_pred = time.time() - t_0_pred
        prediction_per_fold.append(t_f_pred)

        y_pred_true = np.argmax(y_pred, axis = 1)
        y_test_true = np.argmax(y_all[test, :], axis = 1)

        _acc_score = accuracy_score(y_test_true, y_pred_true)
        _f1_score = f1_score(y_test_true, y_pred_true, average = "macro")
        _recall_score = recall_score(y_test_true, y_pred_true, average = "macro")
        _precision_score = precision_score(y_test_true, y_pred_true, average = "macro")

        acc_per_fold.append(_acc_score)
        f1_per_fold.append(_f1_score)
        rec_per_fold.append(_recall_score)
        prec_per_fold.append(_precision_score)

        print('\n{t:=^30}'.format(t = ''))
        print('{m:<10}: {a:.4f}'.format(m = 'Accuracy', a = _acc_score))
        print('{m:<10}: {a:.4f}'.format(m = 'F1 Score', a = _f1_score))
        print('{m:<10}: {a:.4f}'.format(m = 'Recall', a = _recall_score))
        print('{m:<10}: {a:.4f}'.format(m = 'Precision', a = _precision_score))
        print('{t:=^30}\n'.format(t = ''))

    df_scores = pd.DataFrame(columns = ['F1', 'accuracy', 'recall', 'duration', 'precision', 'prediction_time'])
    df_scores['F1'] = f1_per_fold
    df_scores['accuracy'] = acc_per_fold
    df_scores['recall'] = rec_per_fold
    df_scores['duration'] = duration_per_fold
    df_scores['precision'] = prec_per_fold
    df_scores['prediction_time'] = prediction_per_fold

    # Saving scores
    df_scores.to_csv(os.path.join(classifier.output_directory, 'results_k-fold.csv'), index = False)
