import os
import re
import sys
import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine
from train_utils import display_results, display_mean_results


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        Extracts the starting verb in a sentence.
        Args:
            text: str;

        Returns:

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


def load_data(database_filepath):
    """
    Loads available data from a given database path, cleans it and outputs X, Y format.
    Args:
        database_filepath: str; path-like (relative)
    Returns:
        X: array(str); string messages
        Y: array[array(bool)]; message label list
        category_names: list(str)
    """

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)).replace("\\", "/")
    engine = create_engine('sqlite:///' + os.path.join(abs_path, database_filepath))
    df = pd.read_sql_table(database_filepath, engine)

    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = list(Y.columns)

    zero_rows = list(Y[Y.eq(0).all(1)].index)  # get a list of indexes with all labels 0
    bad_ids = np.where((Y != 1) & (Y != 0))  # get a list of indexes with labels different from 0 and 1

    X.drop(list(bad_ids[0]) + zero_rows, axis=0, inplace=True)  # drop the values from X
    Y.drop(list(bad_ids[0]) + zero_rows, axis=0, inplace=True)  # drop the values from Y

    X = X.values
    Y = Y.values

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize one sentence(document) at a time. Applies normalization(alphanumeric), word tokenization, stop-word removal
    and lemmatization(english).
    Args:
        text: str; Sentence-like
    Returns:
        clean_tokens: list(str); list of processed tokens

    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_list = re.findall(url_regex, text)
    for url in url_list:
        text = text.replace(url, " ")

    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in set(stopwords.words("english"))]

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    Build the model pipeline with grid search. Default is CountVectorizer, Tfidf and RandomForestClassifier.
    KNN is another option.
    Returns:
        pipeline: sklearn.GridSearchCV; grid search pipeline
    """

    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(AdaBoostClassifier(), n_jobs=1))
        #("clf", MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
        #("clf", MultiOutputClassifier(KNeighborsClassifier(n_neighbors=36), n_jobs=-1))

    ])

    # pipeline = Pipeline([
    #     ('features', FeatureUnion([
    #
    #         ('text_pipe', Pipeline([
    #             ('vect', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer())
    #         ])),
    #
    #         ('starting_verb', StartingVerbExtractor())
    #     ])),
    #
    #     ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    # ])

    # params = {
    #     # 'features__text_pipe__vect__ngram_range': ((1, 1), (1, 2)),
    #     # 'features__text_pipe__vect__max_df': (0.5, 0.75, 1.0),
    #     # 'features__text_pipe__vect__max_features': (None, 5000, 10000),
    #     # 'features__text_pipe__tfidf__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [50]#, 100, 200],
    #     # 'clf__estimator__min_samples_split': [2, 3, 4],
    #     # 'features__transformer_weights': (
    #     #     {'text_pipeline': 1, 'starting_verb': 0.5},
    #     #     {'text_pipeline': 0.5, 'starting_verb': 1},
    #     #     {'text_pipeline': 0.8, 'starting_verb': 1},
    #     # )
    # }

    params = {
        #'vect__max_df': (0.75, 1.0),
        # 'vect__max_features': (None, 5000, 7500),
        # 'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [100],
        # 'clf__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=params)

    return cv


def evaluate_model(model, X_test, Y_test, category_names, verbose=False):
    """
    Evaluates the trained model pipeline on the test dataset.
    Args:
        model: sklearn.Pipeline;
        X_test: array(str); string messages
        Y_test: array[array(bool)]; message label list
        category_names: list(str); list of category names
        verbose: bool; Flag which enables/disables full output from sklearn.classification_report

    Returns:
        None
    """

    Y_pred = model.predict(X_test)
    report_list = display_results(Y_pred, Y_test, category_names, verbose=verbose)
    display_mean_results(report_list)


def save_model(model, model_filepath):
    """
    Saves the trained CV pipeline to a local path given as argument.
    Args:
        model: sklearn.Pipeline;
        model_filepath: str; Path-like (relative)

    Returns:
        None
    """

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)).replace("\\", "/")
    with open(os.path.join(abs_path, model_filepath), 'wb') as f:
        #joblib.dump(model.best_estimator_, f, compress='zlib')
        joblib.dump(model, f, compress='zlib')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #database_filepath = "data/DisasterResponse.db"
        #model_filepath = "models/classifier.pkl"
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, verbose=False)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
