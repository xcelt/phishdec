# Required libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import re

# Phishing email datasets
DATASET = ['phishing_1.csv', 'validate.csv']
DATASET_2 = ['enron.csv', 'ling.csv', 'spamassassin.csv']

def set_up_data(filename: str, column_names: list, dataset_number: int):
    # Read data from csv file
    df = pd.read_csv(filename, header=0, index_col=0, names=column_names)

    # Identify NaN values
    print("\nCount of NaN values per column:")
    print(df.isna().sum())

    # Option 1: Drop rows where 'email_text' is NaN
    df = df.dropna(subset=['email_text'])

    # # Option 2: Fill/Impute NaN Values
    # df = df.fillna("N/A")
    
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

# Preprocessing: Clean text (removing unwanted characters, emails)
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text

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


def main():
    column_names = ['email_text', 'isPhishing']
    for i in range(len(DATASET)):
        datafile = f'data/{DATASET[i]}'
        dataset_number = i + 1

        df = set_up_data(datafile, column_names, dataset_number)

        phishing_counts = df['isPhishing'].value_counts()
        print(phishing_counts)
        save_graph(data=phishing_counts, title=f'Phishing Email Distribution of {datafile}', ylabel='Number of Emails', filename=f'phish_dist_{dataset_number}.png')

        # Split the dataset into training and testing sets
        X = df['cleaned_email']
        y = df['isPhishing']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_train_counts = y_train.value_counts()
        print(y_train_counts)
        save_graph(data=y_train_counts, title=f'Phishing Email Distribution of {datafile} (Train data)', ylabel='Number of Emails', filename=f'phish_dist_train_{dataset_number}.png')

        y_test_counts = y_test.value_counts()
        print(y_test_counts)
        save_graph(data=y_test_counts, title=f'Phishing Email Distribution of {datafile} (Test data)', ylabel='Number of Emails', filename=f'phish_dist_test_{dataset_number}.png')

        dfs = [phishing_counts, y_train_counts, y_test_counts]
        combined = pd.concat(dfs)
        print(combined)
        save_graph(data=combined, title=f'Phishing Email Distribution of {datafile} (All, train, test data)', ylabel='Number of Emails', filename=f'phish_dist_all_{dataset_number}.png')

        # Convert text data to numerical features using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train the Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test_tfidf)

        # Evaluate the model
        nb_c_matrix = confusion_matrix(y_test, y_pred)
        save_evaluation(y_test, y_pred, model='Naive Bayes', datafile=datafile, c_matrix=nb_c_matrix, dataset_number=dataset_number)

        # Create and train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_tfidf)
        
        # Evaluate the model
        rf_c_matrix = confusion_matrix(y_test, y_pred)
        save_evaluation(y_test, y_pred, model='Random Forest', datafile=datafile, c_matrix=rf_c_matrix, dataset_number=dataset_number)
        
        # Feature importance (for text features only)
        feature_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out()[:1000],
            'importance': rf_model.feature_importances_[:1000]
        })
        print("\nTop 10 Most Important Text Features:")
        print(feature_importance.sort_values('importance', ascending=False).head(10))
 

if __name__ == '__main__':
    main()
