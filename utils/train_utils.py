import re

import nltk
import pandas as pd
import numpy as np
from statistics import mean

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
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


def tokenize(text):
    """
    Tokenize one sentence(document) at a time. Applies normalization(alphanumeric & url), word tokenization,
    stop-word removal and lemmatization(english).
    Args:
        text: str; Sentence-like
    Returns:
        clean_tokens: list(str); list of processed tokens

    """

    # finds and replace urls with an empty space
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_list = re.findall(url_regex, text)
    for url in url_list:
        text = text.replace(url, " ")

    # tokanizes, removes non-alphanumeric characters and stop-words
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in set(stopwords.words("english"))]

    # lemmatizes and strips empty spaces
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        Extracts the starting verb in a sentence.
        Args:
            text: str;

        Returns:
            bool; True if first tag in sentence is verb, False if not
        """

        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))

            try:
                first_word, first_tag = pos_tags[0]
            except IndexError as e:
                first_word, first_tag = None, None

            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)