"""Phishing Email Detection Classification Models"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from utility import save_evaluation

def load_model(filename):
    model = joblib.load(filename)
    return model

def run_nb_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, random_state):
    modelfile = f'model/nb_model_{dataset_number}_{random_state}.sav'
    nb_model = None
    # nb_model = load_model(modelfile)

    if not nb_model:
        # Create and train the Naive Bayes model
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)

    # Predict on the test data
    y_pred = nb_model.predict(X_test_tfidf)

    # Evaluate the model
    test_accuracy = save_evaluation(y_test, y_pred, model='Naive Bayes', datafile=datafile, dataset_number=dataset_number, random_state=random_state)
    
    # Save Naive Bayes model
    joblib.dump(nb_model, modelfile)
    print(f'Naive Bayes model saved to {modelfile}')
    
    return test_accuracy

def run_rf_model(X_train_tfidf, X_test_tfidf, y_train, y_test, datafile, dataset_number, vectorizer, random_state):
    modelfile = f'model/rf_model_{dataset_number}_{random_state}.sav'
    rf_model = None
    # rf_model = load_model(modelfile)

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
    # Feature importance (for text features only)
    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out()[:1000],
        'importance': rf_model.feature_importances_[:1000]
    })
    top_10_features = feature_importance.sort_values('importance', ascending=False).head(10)
    # print("\nTop 10 Most Important Text Features:")
    # print(top_10_features)

    test_accuracy = save_evaluation(y_test, y_pred, model='Random Forest', datafile=datafile, dataset_number=dataset_number, random_state=random_state, imp_features=top_10_features)

    # Save Random Forest model
    joblib.dump(rf_model, modelfile)
    print(f'Random Forest model saved to {modelfile}')

    return test_accuracy
