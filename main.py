"""Phishdec Main App"""

# Required libraries
import sys
import numpy as np
from model import run_nb_model, run_rf_model
from utility import display_welcome_message, preprocess_data

# Phishing email datasets
DATASET = ['phishing_1.csv', 'validate.csv', 'enron.csv', 'ling.csv', 'spamassassin.csv']

def main():
    """Detect Phishing Emails"""
    isRepeated = True
    initialLoad = True
    
    while isRepeated: 
        display_welcome_message()
        menu_option = input('Your option: ')

        if initialLoad:
            for i in range(len(DATASET[:2])):
                datafile = f'data/{DATASET[i]}'
                dataset_number = i + 1
                if i < 1:
                    vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess_data(datafile, dataset_number)
                else:
                    pass
            initialLoad = False

        if menu_option == "0":
            print("Thank you!")
            isRepeated = False
        elif menu_option == "1":
            run_nb_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number)
        elif menu_option == "2":
            run_rf_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, vectorizer)
        else:
            print("Invalid option. Please enter a number shown in the menu.")

if __name__ == '__main__':
    sys.exit(main())
    

        
        
