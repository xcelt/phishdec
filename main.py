# Required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import re

# Phishing email datasets
DATASETS = ['data/phishing_1.csv']
VALIDATION_DATASET = 'data/validate.csv'
PHISHING_DISTRIBUTION = 'output/phishing_distribution.png'
PHISHING_DISTRIBUTION_TRAIN = 'output/phishing_distribution_train.png'
PHISHING_DISTRIBUTION_TEST = 'output/phishing_distribution_test.png'
PHISHING_DISTRIBUTION_ALL = 'output/phishing_distribution_all.png'
CONFUSION_MATRIX = 'output/confusion_matrix.png'

# Preprocessing: Clean text (removing unwanted characters, emails)
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text

def main():
    column_names = ['email_text', 'isPhishing']
    for i in range(len(DATASETS)):
        data = DATASETS[i]

        # Read data from csv file
        df = pd.read_csv(data, header=0, index_col=0, names=column_names)
        # df_validate = pd.read_csv(VALIDATION_DATASET, header=0, index_col=0, names=column_names)

        # Drop rows where 'email_text' is NaN
        df = df.dropna(subset=['email_text'])
        
        df['id'] = range(1, len(df) + 1)
        
        # Reorder id column
        df = df[['id', 'email_text', 'isPhishing']]

        # Map data isPhishing label: 1 for phishing, 0 for non-phishing
        df['isPhishing'] = df['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})
        # df_validate['isPhishing'] = df_validate['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})

        # print(df.head())
        # print("\nCount of NaN values per column:")
        # print(df.isna().sum())
        phishing_counts = df['isPhishing'].value_counts()
        print(phishing_counts)
        
        phishing_counts.plot(kind='bar', title='Phishing Email Distribution')
        plt.ylabel(f'Number of Emails')
        plt.savefig(PHISHING_DISTRIBUTION)
        plt.close()
        print('Bar chart saved to', PHISHING_DISTRIBUTION)
        # print(df_validate['isPhishing'].value_counts())

        df['cleaned_email'] = df['email_text'].apply(preprocess_text)
        # print(df.head())
        df_clean = df[['id', 'cleaned_email', 'isPhishing']]

        # Split the dataset into training and testing sets
        X = df['cleaned_email']
        y = df['isPhishing']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_train_counts = y_train.value_counts()
        print(y_train_counts)
        y_train_counts.plot(kind='bar', title='Phishing Email Distribution (Train data)')
        plt.ylabel(f'Number of Emails')
        plt.savefig(PHISHING_DISTRIBUTION_TRAIN)
        plt.close()
        print('Bar chart (train data) saved to', PHISHING_DISTRIBUTION_TRAIN)
        y_test_counts = y_test.value_counts()
        print(y_test_counts)
        y_test_counts.plot(kind='bar', title='Phishing Email Distribution (Test data)')
        plt.ylabel(f'Number of Emails')
        plt.savefig(PHISHING_DISTRIBUTION_TEST)
        plt.close()
        print('Bar chart (test data) saved to', PHISHING_DISTRIBUTION_TEST)

        dfs = [phishing_counts, y_train_counts, y_test_counts]
        combined = pd.concat(dfs)
        print(combined)
        combined.plot(kind='bar', title='Phishing Email Distribution (All, train, test data')
        plt.ylabel(f'Number of Emails')
        plt.savefig(PHISHING_DISTRIBUTION_ALL)
        plt.close()
        print('Bar chart (combined) saved to', PHISHING_DISTRIBUTION_ALL)

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
        accuracy = accuracy_score(y_test, y_pred)
        c_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print('Classification Report:')
        print(report)

        # # Validate data and model
        # z_pred = df_validate.predict(X_test_tfidf)
        # accuracy_v = accuracy_score(y_test, z_pred)
        # report_v = classification_report(y_test, z_pred)
        # print(f'Accuracy (Validate): {accuracy_v:.4f}')
        # print('Classification Report (Validate):')
        # print(report_v)

        display = ConfusionMatrixDisplay(c_matrix)
        display.plot()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(CONFUSION_MATRIX)
        print('Confusion matrix saved to', CONFUSION_MATRIX)

        dataset_number = i + 1
        clean_data = f'data/clean_{dataset_number}.csv'
        # clean_validate_data = f'data/clean_validate_{dataset_number}.csv'

        # Save clean data to csv file
        df_clean.to_csv(clean_data, index=False)
        # df_validate.to_csv(clean_validate_data, index=False)
        print('Clean data saved to', clean_data)
        # print('Clean validate data saved to', clean_validate_data)

        result = f'output/model_evaluation_{dataset_number}.txt'
        
        # Output results to a text file
        with open(result, 'w') as file:
            file.write(f'Accuracy: {accuracy:.4f}\n')
            file.write('\nClassification Report:\n')
            file.write(report)

        print('Evaluation report saved to', result)

        

if __name__ == '__main__':
    main()
