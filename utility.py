"""Utility module"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import t as t_dist
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing: Clean text (removing unwanted characters, emails)
def preprocess_text(text):
    if type(text) != int:
        text = text.lower()  # Lowercase the text
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text

def preprocess_data(df: pd.DataFrame, dataset_number: int):
    # Identify NaN values
    print("\nCount of NaN values per column:")
    print(df.isna().sum())

    if dataset_number in range(1, 3):
        # # Option 1: Drop rows where value is NaN
        # df = df.dropna(subset=['email_text'])

        # Option 2: Fill/Impute NaN Values
        df = df.fillna("N/A")
        
        df['id'] = range(1, len(df) + 1)
        
        # Reorder id column
        df = df[['id', 'email_text', 'isPhishing']]

        # Map data isPhishing label: 1 for phishing, 0 for non-phishing
        df['isPhishing'] = df['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})

        df['clean_email'] = df['email_text'].apply(preprocess_text)
        df_clean = df[['id', 'clean_email', 'isPhishing']]
    elif dataset_number in range (3, 5):
        # # Option 1: Drop rows where value is NaN
        # df = df.dropna(subset=['subject', 'body'])

        # Option 2: Fill/Impute NaN Values
        df = df.fillna("N/A")
        
        df['id'] = range(1, len(df) + 1)
        
        # Reorder id column
        df = df[['id', 'subject', 'body', 'label']]

        # Rename columns
        df = df.rename(columns={'label': 'isPhishing'})

        df['clean_subject'] = df['subject'].apply(preprocess_text)
        df['clean_body'] = df['body'].apply(preprocess_text)
        df_clean = df[['id', 'clean_subject', 'clean_body', 'isPhishing']]
    elif dataset_number in range (5, 10):
        # # Option 1: Drop rows where value is NaN
        # df = df.dropna(subset=['sender', 'receiver', 'date', 'subject', 'body', 'urls'])

        # Option 2: Fill/Impute NaN Values
        df = df.fillna("N/A")
        
        df['id'] = range(1, len(df) + 1)
        
        # Reorder id column
        df = df[['id', 'sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']]

        # Rename columns
        df = df.rename(columns={'label': 'isPhishing'})

        df['clean_sender'] = df['sender'].apply(preprocess_text)
        df['clean_receiver'] = df['receiver'].apply(preprocess_text)
        df['clean_date'] = df['date'].apply(preprocess_text)
        df['clean_subject'] = df['subject'].apply(preprocess_text)
        df['clean_body'] = df['body'].apply(preprocess_text)
        df['clean_urls'] = df['urls'].apply(preprocess_text)
        df_clean = df[['id', 'clean_sender', 'clean_receiver', 'clean_date', 'clean_subject', 
                       'clean_body', 'isPhishing', 'clean_urls']]
    
    # # Save clean data to csv file
    # clean_data = f'data/clean_{dataset_number}.csv'
    # df_clean.to_csv(clean_data, index=False)
    # print('Clean data saved to', clean_data)

    return df_clean

def prepare_data(datafile: str, dataset_number: int):
    if dataset_number in range(1, 3):
        if 'clean' not in datafile:
            column_names = ['email_text', 'isPhishing']
            # Read data from csv file
            df = pd.read_csv(datafile, header=0, index_col=0, names=column_names)
        else:
            df = pd.read_csv(datafile, header=0, index_col=0)
    elif dataset_number in range (3, 5):
        # Read data from csv file
        df = pd.read_csv(datafile, header=0)
    elif dataset_number in range (5, 10):
        # Read data from csv file
        df = pd.read_csv(datafile, header=0, lineterminator='\n')
    
    if 'clean' not in datafile:
        # Preprocess and save clean data to file
        df = preprocess_data(df, dataset_number)  

    return df

def split_data(datafile: str, dataset_number: int, df: pd.DataFrame, test_size: float, random_state: int):
    phishing_counts = df['isPhishing'].value_counts()
    # print(phishing_counts)
    save_graph(data=phishing_counts, title=f'Phishing Email Distribution of {datafile}', ylabel='Number of Emails', filename=f'phish_dist_{dataset_number}_{random_state}.png')

    if dataset_number in range(1, 3):
        X = df['clean_email']
    elif dataset_number in range (3, 10):
        X = df['clean_body']

    y = df['isPhishing']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    y_train_counts = y_train.value_counts()
    # print(y_train_counts)
    save_graph(data=y_train_counts, title=f'Phishing Email Distribution of {datafile} (Train data)', ylabel='Number of Emails', filename=f'phish_dist_train_{dataset_number}_{random_state}.png')

    y_test_counts = y_test.value_counts()
    # print(y_test_counts)
    save_graph(data=y_test_counts, title=f'Phishing Email Distribution of {datafile} (Test data)', ylabel='Number of Emails', filename=f'phish_dist_test_{dataset_number}_{random_state}.png')

    dfs = [phishing_counts, y_train_counts, y_test_counts]
    combined = pd.concat(dfs)
    # print(combined)
    save_graph(data=combined, title=f'Phishing Email Distribution of {datafile} (All, train, test data)', ylabel='Number of Emails', filename=f'phish_dist_all_{dataset_number}_{random_state}.png')

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test

def save_graph(data: pd.Series, title: str, ylabel: str, filename: str):
    data.plot(kind='bar', title=title)
    plt.ylabel(ylabel)
    filename = f'graph/' + filename
    plt.savefig(filename)
    plt.close()
    print(f'{title} bar graph saved to {filename}')

def save_confusion_matrix(c_matrix, model: str, datafile: str, dataset_number: int, random_state: int):
    model_initials = 'nb' if model == "Naive Bayes" else 'rf'
    cmatrixfile = f'graph/{model_initials}_c_matrix_{dataset_number}_{random_state}.png'
    display = ConfusionMatrixDisplay(c_matrix)
    display.plot()
    plt.title(f'Confusion Matrix of {model} model for {datafile}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(cmatrixfile)
    plt.close()
    print(f'Confusion matrix of {model} model for {datafile} saved to {cmatrixfile}')
    
def save_evaluation(y_test, y_pred, model: str, datafile: str, dataset_number: int, random_state: int, imp_features: pd.DataFrame = None):
    accuracy = accuracy_score(y_test, y_pred)
    c_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    # print("Confusion Matrix:")
    # print(c_matrix)
    # print('Classification Report:')
    # print(report)

    save_confusion_matrix(c_matrix=c_matrix, model=model, datafile=datafile, dataset_number=dataset_number, random_state=random_state)
    
    model_initials = 'nb' if model == "Naive Bayes" else 'rf'
    reportfile = f'result/{model_initials}_evaluation_{dataset_number}_{random_state}.txt'
    # Output results to a text file
    with open(reportfile, 'w') as file:
        file.write(f'Dataset: {datafile}\n')
        file.write(f'Model: {model}\n')
        file.write(f'Accuracy: {accuracy:.4f}\n')
        file.write(f'\nConfusion Matrix:\n')
        file.write(f'True Positive (TP)     False Negative (FN)\n')
        file.write(f'False Positive (FP)     True Negative (TN)\n')
        file.write(str(c_matrix))
        file.write('\n\nClassification Report:\n')
        file.write(report)

        if model_initials == 'rf':
            file.write(f'\nTop 10 Most Important Text Features:\n')
            file.write(str(imp_features))

    print('Evaluation report saved to', reportfile)
    
    return accuracy

def five_two_statistic(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p_hat = (p1 + p2) / 2
    s = (p1 - p_hat)**2 + (p2 - p_hat)**2
    t = p1[0] / np.sqrt(1/5. * sum(s))

    p_value = t_dist.sf(t, 5)*2

    return t, p_value

def save_cv_test(datafile: str, dataset_number: int, p_1: list, p_2: list):
    print('5x2 CV Paired t-test')     
    t, p = five_two_statistic(p_1, p_2)
    print(f't statistic: {t}, p-value: {p}n')

    resultfile = f'result/cv_paired_t_test_{dataset_number}.txt'
    # Output results to a text file
    with open(resultfile, 'w') as file:
        file.write(f'5x2 Cross-Validation Paired t-test\n')
        file.write(f'Dataset: {datafile}\n')
        file.write(f't statistic: {t}, p-value: {p}n')

    print('CV Paired t-test results saved to', resultfile)
    
def display_welcome_message():
    print(f"""
    ============ Version 0.1.0 ============ 
    Welcome to the Phishdec menu    
    ======================================= \n
    Choose Your Option:
    1. Evaluate Naive Bayes model
    2. Evaluate Random Forest model
    3. 5×2 Cross-Validation Test (both models)
    0. Quit
    """)
    