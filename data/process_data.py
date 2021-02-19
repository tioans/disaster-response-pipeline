import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges two csv files, containing messages and categories respectively.
    :param messages_filepath: str; path-like
    :param categories_filepath: str; path-like
    :return: pd.DataFrame;
    """

    # read data from csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on="id", how="inner")


def clean_data(df):
    """
    Cleans the joined dataframe by expanding&converting the categories column, and removes duplicates.
    :param df: pd.DataFrame; messages and categories merged
    :return: df: pd.DataFrame;
    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    row = categories.iloc[0, :]  # select the first row of the categories dataframe
    category_colnames = row.apply(lambda x: str(x)[:-2])  # extract only the column names from first row
    categories.columns = category_colnames  # rename the columns of `categories`

    # convert category values to ints
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    df.drop(["categories"], axis=1, inplace=True)  # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1)  # concatenate the original dataframe with the new `categories` dataframe

    # check and remove duplicates
    print("Num. duplicates before removing: ", df.duplicated(subset='id', keep='first').sum())
    df = df.drop_duplicates(subset=["id"], keep="first")  # drop duplicates
    print("Num. duplicates after removing : ", df.duplicated(subset='id', keep='first').sum())

    # check for zero rows and rows with labels different from 1 or 0
    zero_rows = list(df[df[categories.columns].eq(0).all(1)].index)
    bad_ids = np.where((df[categories.columns] != 1) & (df[categories.columns] != 0))

    # keep only rows which are not in the lists above. X-files?!
    print("Num. bad rows before removing : ", df.shape[0])
    idx_to_keep = set(range(df.shape[0])) - set(list(bad_ids[0]) + zero_rows)
    df = df.take(list(idx_to_keep))
    print("Num. bad rows after removing  : ", df.shape[0])

    return df


def save_data(df, database_filename):
    """
    Save a dataframe to an SQL Lite DB with the sqlalchemy engine.
    :param df: pd.DataFrame
    :param database_filename: str; path with name for the SQL lite DB
    :return: None
    """

    # create DB connection and save table
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()