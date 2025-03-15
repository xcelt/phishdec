
"""Utility module."""

import re
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing: Clean text (removing unwanted characters, emails)
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text

def set_up_data(filename: str, column_names: list, dataset_number: int):
    # Read data from csv file
    df = pd.read_csv(filename, header=0, index_col=0, names=column_names)

    # # Identify NaN values
    # print("\nCount of NaN values per column:")
    # print(df.isna().sum())

    # # Option 1: Drop rows where 'email_text' is NaN
    # df = df.dropna(subset=['email_text'])

    # Option 2: Fill/Impute NaN Values
    df = df.fillna("N/A")
    
    df['id'] = range(1, len(df) + 1)
    
    # Reorder id column
    df = df[['id', 'email_text', 'isPhishing']]

    # Map data isPhishing label: 1 for phishing, 0 for non-phishing
    df['isPhishing'] = df['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})

    df['cleaned_email'] = df['email_text'].apply(preprocess_text)
    df_clean = df[['id', 'cleaned_email', 'isPhishing']]
    clean_data = f'data/clean_{dataset_number}.csv'

    # Save clean data to csv file
    df_clean.to_csv(clean_data, index=False)
    print('Clean data saved to', clean_data)

    return df

def preprocess_data(datafile: str, dataset_number: int):
    column_names = ['email_text', 'isPhishing']

    df = set_up_data(datafile, column_names, dataset_number)

    phishing_counts = df['isPhishing'].value_counts()
    # print(phishing_counts)
    save_graph(data=phishing_counts, title=f'Phishing Email Distribution of {datafile}', ylabel='Number of Emails', filename=f'phish_dist_{dataset_number}.png')

    # Split the dataset into training and testing sets
    X = df['cleaned_email']
    y = df['isPhishing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train_counts = y_train.value_counts()
    # print(y_train_counts)
    save_graph(data=y_train_counts, title=f'Phishing Email Distribution of {datafile} (Train data)', ylabel='Number of Emails', filename=f'phish_dist_train_{dataset_number}.png')

    y_test_counts = y_test.value_counts()
    # print(y_test_counts)
    save_graph(data=y_test_counts, title=f'Phishing Email Distribution of {datafile} (Test data)', ylabel='Number of Emails', filename=f'phish_dist_test_{dataset_number}.png')

    dfs = [phishing_counts, y_train_counts, y_test_counts]
    combined = pd.concat(dfs)
    # print(combined)
    save_graph(data=combined, title=f'Phishing Email Distribution of {datafile} (All, train, test data)', ylabel='Number of Emails', filename=f'phish_dist_all_{dataset_number}.png')

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test

def save_graph(data: pd.Series, title: str, ylabel: str, filename: str):
    data.plot(kind='bar', title=title)
    plt.ylabel(ylabel)
    filename = f'output/' + filename
    plt.savefig(filename)
    plt.close()
    print(f'{title} bar graph saved to {filename}')

def save_confusion_matrix(c_matrix, model: str, datafile: str, dataset_number: int):
    model_initials = 'nb' if model == "Naive Bayes" else 'rf'
    filename = f'output/{model_initials}_c_matrix_{dataset_number}.png'
    display = ConfusionMatrixDisplay(c_matrix)
    display.plot()
    plt.title(f'Confusion Matrix of {model} model for {datafile}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()
    print(f'Confusion matrix of {model} model for {datafile} saved to {filename}')
    
def save_evaluation(y_test, y_pred, model: str, datafile: str, c_matrix, dataset_number: int):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print("Confusion Matrix:")
    print(c_matrix)
    print('Classification Report:')
    print(report)

    save_confusion_matrix(c_matrix=c_matrix, model=model, datafile=datafile, dataset_number=dataset_number)
    
    model_initials = 'nb' if model == "Naive Bayes" else 'rf'
    filename = f'output/{model_initials}_evaluation_{dataset_number}.txt'
    # Output results to a text file
    with open(filename, 'w') as file:
        file.write(f'Dataset: {datafile}\n')
        file.write(f'Model: {model}\n')
        file.write(f'Accuracy: {accuracy:.4f}\n')
        file.write('\nClassification Report:\n')
        file.write(report)
    print('Evaluation report saved to', filename)

def display_welcome_message():
    print(f"""
    ============ Version 0.1.0 ============ 
    Welcome to the Phishdec menu
    ======================================= \n
    Choose Your Option:
    1. Evaluate Naive Bayes model
    2. Evaluate Random Forest model
    0. Quit
    """)
    