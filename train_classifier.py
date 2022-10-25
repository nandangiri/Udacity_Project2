import sys
import pandas as pd
import numpy as np
import warnings

from sqlalchemy import create_engine

import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download(['wordnet', 'punkt', 'stopwords']) 


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Input:
        database_filepath: the file path of the database
    Output:
        X: independent_var
        Y: dependent_var
        category_names: category of dependent_var
    """
    clean_engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("cleaned_table001", con=clean_engine)
    df.dropna(axis=0, subset=df.columns[4:], inplace=True)
    X = df["message"]  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Input:
        text: message
    Output: 
        a list of lemmed words and normalized token
    """
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenization
    words = word_tokenize(text)
   
    # normalization word tokens and remove stop words
    stopwords_cleaned = stopwords.words("english")
    normlizer = PorterStemmer()
    
    normlized = [normlizer.stem(word) for word in words if word not in stopwords_cleaned]
    return normlized
    

def build_model():
    """
    Input: N/A
    Ouput: classifier
    """
    # Pipleine: Random Forest Classifier
    pipeline_cleaned = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Using grid search
    # Create Grid search parameters for Random Forest Classifier   
    parameters_cleaned = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 40]
    }

    model_cleaned = GridSearchCV(pipeline_cleaned, param_grid = parameters_cleaned)
    return model_cleaned

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input: 
        model: model 
        X_test: test set of independent_var
        Y_test: test set of dependent_var
        category_names: category names
    Output: NA
    """
    
    Y_prediction = model.predict(X_test)

    i = 0
    for col in Y_test:
        print(classification_report(Y_test[col], Y_prediction[:, i]))
        i += 1
    accuracy = (Y_prediction == Y_test.values).mean()
    print("Accuracy of Model {:.3f}".format(accuracy))
    return


def save_model(model, model_filepath):
    """
    Input:
        model : model
        model_filepath: destination
    Output: NA
    """
    file_name = model_filepath
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()