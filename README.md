# Disaster Response Pipeline Project

### Project Introduction
We are building an ML model to classify text messages during a disaster. The Dataset is given by Appen. We classify the data into different categories, so that the messages can be allocated to the respective entities. The model result is a visual presented on a web app.

### Files in the Repository Description
app folder: package for running web app.
data folder: the data input tables (disaster_categories.csv, disaster_messages.csv), the SQL database (DisasterResponse.db), the code that loads and processes the csv input to create a database (process_data.py).
models folder: the machine learning model (train_classifier.py). The result generated (classifier.pkl) is too large for github upload. So, please refer the workspace.

### Files in the Repository Details
app

- template
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

data

- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # data cleaning pipeline
- disasterresponse.db # database to save clean data to

models

- train_classifier.py # machine learning pipeline
- classifier.pkl # saved model

README.md

1. ETL Pipeline
A Python script, process_data.py, writes a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
A jupyter notebook ETL Pipeline Preparation was used to do EDA to prepare the process_data.py python script.

2. ML Pipeline
A Python script, train_classifier.py, writes a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
A jupyter notebook ML Pipeline Preparation was used to do EDA to prepare the train_classifier.py python script.

3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


### Instructions to execute the files and commands
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage
