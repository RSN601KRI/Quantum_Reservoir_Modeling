from sklearn.metrics import recall_score, accuracy_score

def evaluate_model(classifier, X_test, y_test):
    predictions = classifier.predict(X_test)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    return {"Recall": recall, "Accuracy": accuracy}
