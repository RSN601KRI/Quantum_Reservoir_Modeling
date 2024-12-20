# main.py
from quantum_reservoir_computing import quantum_model, train_model
from data_preprocessing import load_and_preprocess_data
from model_evaluation import evaluate_model

# Load and preprocess data
X_train, y_train = load_and_preprocess_data('credit_risk_data.csv')

# Train your quantum reservoir model
model = train_model(X_train, y_train)

# Evaluate the model
X_test, y_test = load_and_preprocess_data('credit_risk_data_test.csv')
evaluate_model(model, X_test, y_test)
