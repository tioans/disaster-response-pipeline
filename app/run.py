import json
import os
import sys
import nltk
import plotly
import joblib
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from utils.train_utils import StartingVerbExtractor, tokenize

app = Flask(__name__)

# load data
database_filepath = os.path.join("data", "DisasterResponse.db")
model_filepath = os.path.join("models", "classifier.pkl")
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
engine = create_engine('sqlite:///' + os.path.join(abs_path, database_filepath))
df = pd.read_sql_table("data/DisasterResponse.db", engine)

# load model
model = joblib.load(os.path.join(abs_path, model_filepath))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract and calculate categories and number of occurrences
    category_names = df.iloc[:, 4:].columns
    category_flags = (df.iloc[:, 4:] == 1).sum().values

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },

        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_flags
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
