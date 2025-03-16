"""Phishdec Main App"""

# Required libraries
import sys
import numpy as np
import pandas as pd
from model import run_nb_model, run_rf_model
from utility import display_welcome_message, prepare_data, save_cv_test, split_data

# Phishing email datasets
RAW_DATASET = ['phishing.csv', 'validate.csv', 'enron.csv', 'ling.csv', 
               'spamassassin.csv', 'ceas_08.csv', 'trec_05.csv', 'trec_06.csv', 'trec_07.csv']

DATASET_LENGTH = len(RAW_DATASET)

def two_fold_cv(datafile: str, dataset_number: int, df: pd.DataFrame, test_size: float):
    p_1 = []
    p_2 = []

    rng = np.random.RandomState(42)
    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = split_data(datafile, dataset_number, df, test_size=test_size, random_state=randint)

        test_accuracy_nb = run_nb_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, random_state=randint)
        test_accuracy_rf = run_rf_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, vectorizer, random_state=randint)
        p_1.append(test_accuracy_nb - test_accuracy_rf)

        train_accuracy_nb = run_nb_model(X_test_tfidf, X_train_tfidf, y_test, y_train, datafile, dataset_number, random_state=randint)
        train_accuracy_rf = run_rf_model(X_test_tfidf, X_train_tfidf, y_test, y_train, datafile, dataset_number, vectorizer, random_state=randint)
        p_2.append(train_accuracy_nb - train_accuracy_rf)
    
    # Running the 5×2 Cross-Validation Test
    save_cv_test(datafile, dataset_number, p_1, p_2)

def evaluate_model_all_dataset(option: str):
    for i in range(DATASET_LENGTH):
        # Option 1 (stable): Load Raw data
        datafile = f'data/{RAW_DATASET[i]}'

        dataset_number = i + 1
        # Option 2 (alpha test): Load clean data
        # datafile = f'data/clean_{dataset_number}.csv'

        df = prepare_data(datafile, dataset_number)
        
        if option == "1":
            vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = split_data(datafile, dataset_number, df, test_size=0.2, random_state=42)
            test_accuracy_nb = run_nb_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, random_state=42)
        elif option == "2":
            vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = split_data(datafile, dataset_number, df, test_size=0.2, random_state=42)
            test_accuracy_rf = run_rf_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, vectorizer, random_state=42)
        elif option == "3":
            two_fold_cv(datafile, dataset_number, df, test_size=0.5)

def main():
    """Detect Phishing Emails"""
    isRepeated = True
    
    while isRepeated: 
        display_welcome_message()
        menu_option = input('Your option: ')

        if menu_option == "0":
            print("Thank you!")
            isRepeated = False
        elif menu_option == "1" or menu_option == "2" or menu_option == "3":
            evaluate_model_all_dataset(menu_option)
        else:
            print("Invalid option. Please enter a number shown in the menu.")

if __name__ == '__main__':
    sys.exit(main())
    

        
        
