import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv("./credit_risk.csv")

    # Separate features and target
    X = data.drop(columns=["Default"])
    y = data["Default"]

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
