"""Phishing Email Detection Classification Models"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from utility import save_evaluation

def load_model(filename):
    model = joblib.load(filename)
    return model

def run_nb_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number):
    filename = f'model/nb_model.sav'
    nb_model = None
    nb_model = load_model(filename)

    if not nb_model:
        # Create and train the Naive Bayes model
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)

    # Predict on the test data
    y_pred = nb_model.predict(X_test_tfidf)

    # Evaluate the model
    nb_c_matrix = confusion_matrix(y_test, y_pred)
    save_evaluation(y_test, y_pred, model='Naive Bayes', datafile=datafile, c_matrix=nb_c_matrix, dataset_number=dataset_number)
    
    # Save Naive Bayes model
    joblib.dump(nb_model, filename)
    print(f'Naive Bayes model saved to {filename}')

def run_rf_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, vectorizer):
    filename = f'model/rf_model.sav'
    rf_model = None
    rf_model = load_model(filename)

    if not rf_model:
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

    # Save Random Forest model
    joblib.dump(rf_model, filename)
    print(f'Random Forest model saved to {filename}')
