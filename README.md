# Disaster Response Pipeline Project

### Project Introduction
We are building an ML model to classify text messages during a disaster. The Dataset is given by Appen. We classify the data into different categories, so that the messages can be allocated to the respective entities. The model result is a visual presented on a web app.

### Files in the Repository Description
app folder: package for running web app.
data folder: the data input tables (disaster_categories.csv, disaster_messages.csv), the SQL database (DisasterResponse.db), the code that loads and processes the csv input to create a database (process_data.py).
models folder: the machine learning model (train_classifier.py). The result generated (classifier.pkl) is too large for github upload. So, please refer the workspace.

### Files in the Repository Details
1. app folder
- template
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

2. data folder
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py
- InsertDatabaseName.db # database to save clean data to

3. models folder
- train_classifier.py
- classifier.pkl # saved model

4. README.md


### Instructions to execute the files and commands
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage
