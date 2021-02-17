import pandas as pd
import numpy as np
from statistics import mean

from sklearn.metrics import classification_report


def calc_accuracy(y_test, y_pred, digit_prec=4):
    """
    Calculates the accuracy metric for classification task.
    Args:
        y_test: array[array(bool)]; true labels
        y_pred: array[array(bool)]; predicted labels
        digit_prec: int; number of digits (precision)

    Returns:
        int; rounded accuracy values
    """

    return round((y_test == y_pred).mean(), digit_prec)


def display_results(y_pred, y_test, category_names, verbose=False, digit_prec=4):
    """
    Display classification metrics. If verbose=True, display full results for each category. Optionally, returns list
    of report objects for each class (either str or dict, depending on the verbose flag).
    Args:
        y_pred: array(array(bool));
        y_test: array(array(bool));
        category_names: list(str); list of category names
        verbose: bool; Flag
        digit_prec: int; number of digits (precision)

    Returns:
        report_list: list(sklearn.classification_report); Optional, can be ignored.
    """

    y_pred_df = pd.DataFrame(y_pred)
    y_test_df = pd.DataFrame(y_test)

    report_list = []
    for i in range(len(category_names)):

        if verbose:
            report = classification_report(y_test_df.iloc[:, i], y_pred_df.iloc[:, i], labels=np.unique(y_pred),
                                           output_dict=False)
            acc = calc_accuracy(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
            print("Class: ", category_names[i])
            print(report)
        else:
            report = classification_report(y_test_df.iloc[:, i], y_pred_df.iloc[:, i], labels=np.unique(y_pred),
                                           output_dict=True)
            try:
                acc = report["accuracy"]
            except Exception as e:
                acc = calc_accuracy(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])

            print("Class: {} -> Acc: {}; Prec: {}; Rec: {};".format(category_names[i], round(acc, digit_prec),
                                                       round(report["weighted avg"]["precision"], digit_prec),
                                                       round(report["weighted avg"]["recall"], digit_prec)))

        report_list.append((report, acc))

    return report_list


def display_mean_results(report_list, digit_prec=4):
    """
    Displays mean accuracy, precision and recall scores for a model based on the individual class scores.
    Args:
        report_list: list(sklearn.classification_report); classification_report should be a dict
        digit_prec: int; number of digits (precision)

    Returns:
        None
    """

    acc_list, prec_list, rec_list = [], [], []
    for report in report_list:
        acc_list.append(report[1])
        prec_list.append(report[0]["weighted avg"]["precision"])
        rec_list.append(report[0]["weighted avg"]["recall"])

    print("\nModel: \n Acc: {}; Prec: {}; Rec: {};".format(round(mean(acc_list), digit_prec),
                                                           round(mean(prec_list), digit_prec),
                                                           round(mean(rec_list), digit_prec)))
