import os
import pandas as pd
from preprocess import preprocess_data
from quantum_reservoir import QuantumReservoir
from classifier import train_classifier
from evaluation import evaluate_model

if __name__ == "__main__":

    # Paths and constants
    data = pd.read_csv("./credit_risk.csv")


    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data("./credit_risk.csv")

    # Initialize quantum reservoir
    reservoir = QuantumReservoir(num_qubits=4, num_layers=3)

    # Transform data using the quantum reservoir
    reservoir_outputs_train = [reservoir.run(data) for data in X_train]
    reservoir_outputs_test = [reservoir.run(data) for data in X_test]

    # Train the classifier
    classifier = train_classifier(reservoir_outputs_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(classifier, reservoir_outputs_test, y_test)

    # Print results
    print(f"Model Evaluation:\nRecall: {metrics['Recall']}\nAccuracy: {metrics['Accuracy']}")

