import sys
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input: 
    messages_filepath: message table
    categories_filepath: category table
    
    Output: df: merged table
    """
    # crate message table
    messages_table = pd.read_csv(messages_filepath)
    messages_table.head()
    
    # create categories table
    categories_table = pd.read_csv(categories_filepath)
    categories_table.head()

    # merge tables on column ID
    df = messages_table.merge(categories_table, how="left", on='id')
    df.head()
    
    return df
    

def clean_data(df):
    """
    Input: df: source_data
    Output: df: cleaned_data
    """
    # create a df of 36 individual category columns
    categories_table = df['categories'].str.split(';', expand=True)
    categories_table.head()

    # select the first row of the categories dataframe
    rows = categories_table.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    categorycol = rows.applymap(lambda x: x[:-2]).iloc[0,:]
    
    
    # rename the columns of `categories`
    categories_table.columns = categorycol
   
    
    # convert category values to 0 / 1 numerics
    for column in categories_table:
        # set each value to be the last character of the string
        categories_table[column] = categories_table[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories_table[column] = categories_table[column].astype(int) 
      
    categories_table = categories_table[categories_table["related"] < 2]
    # drop the original categories column from `df`
    df = df.drop(['categories'],axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories_table], axis = 1 )
   
    # drop duplicates
    df_cleaned = df.drop_duplicates()
  
    return df_cleaned
    

def save_data(df, database_filename):
    """
    Input: 
        df : dataframe
        database_filename: destination in sqlite
    Output:
        NA
    """
    clean_engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("cleaned_table001", clean_engine, index=False, if_exists="replace")


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