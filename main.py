# Required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re

# Phishing email datasets
DATASETS = ['data/phishing_1.csv']
VALIDATION_DATASET = 'data/validate.csv'

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
        df_validate = pd.read_csv(VALIDATION_DATASET, header=0, names=column_names)

        # Drop rows where 'email_text' is NaN
        df = df.dropna(subset=['email_text'])
        df_validate = df_validate.dropna(subset=['email_text'])
        
        df['id'] = range(1, len(df) + 1)
        
        # Reorder id column
        df = df[['id', 'email_text', 'isPhishing']]

        # Map data isPhishing label: 1 for phishing, 0 for non-phishing
        df['isPhishing'] = df['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})
        df_validate['isPhishing'] = df_validate['isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})

        print(df.head())
        # print("\nCount of NaN values per column:")
        # print(df.isna().sum())
        print(df['isPhishing'].value_counts())
        print()
        print(df_validate['isPhishing'].value_counts())

        df['cleaned_email'] = df['email_text'].apply(preprocess_text)
        print(df.head())

        # Split the dataset into training and testing sets
        X = df['cleaned_email']
        y = df['isPhishing']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        report = classification_report(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print('Classification Report:')
        print(report)

        # Validate data and model
        z_pred = df_validate.predict(X_test_tfidf)
        accuracy_v = accuracy_score(y_test, z_pred)
        report_v = classification_report(y_test, z_pred)
        print(f'Accuracy (Validate): {accuracy_v:.4f}')
        print('Classification Report (Validate):')
        print(report_v)

        dataset_number = i + 1
        clean_data = f'data/clean_{dataset_number}.csv'
        clean_validate_data = f'data/clean_validate_{dataset_number}.csv'

        # Save clean data to csv file
        df.to_csv(clean_data, index=False)
        df_validate.to_csv(clean_validate_data, index=False)
        print('Clean data saved to', clean_data)
        print('Clean validate data saved to', clean_validate_data)

        result = f'output/model_evaluation_{dataset_number}.txt'
        
        # Output results to a text file
        with open(result, 'w') as file:
            file.write(f'Accuracy: {accuracy:.4f}\n')
            file.write('\nClassification Report:\n')
            file.write(report)

            file.write(f'Accuracy (Validate): {accuracy_v:.4f}\n')
            file.write('\nClassification Report (Validate):\n')
            file.write(report_v)

        print('Output written to', result)

        

if __name__ == '__main__':
    main()


# # Sample prediction
# sample_email = "Congratulations, you've won a free iPhone! Claim your prize now."
# processed_sample_email = preprocess_text(sample_email)
# sample_tfidf = vectorizer.transform([processed_sample_email])
# prediction = model.predict(sample_tfidf)

# if prediction == 1:
#     print("The email is Phishing.")
# else:
#     print("The email is Not Phishing.")