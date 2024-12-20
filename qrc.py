import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import recall_score

# Load the dataset (modify file path as needed)
data_path = "german.data" 

# Load the dataset (modify file path as needed)
data_path = "german.data"  # or "german.data-numeric"
column_names = [
    "Status_checking", "Duration", "Credit_history", "Purpose", "Credit_amount", "Savings_account",
    "Employment", "Installment_rate", "Personal_status", "Other_debtors", "Present_residence",
    "Property", "Age", "Other_installment_plans", "Housing", "Existing_credits", "Job", "Number_liable",
    "Telephone", "Foreign_worker", "Class"
]

# Load dataset into a DataFrame
df = pd.read_csv(data_path, header=None, names=column_names, delim_whitespace=True)

# Handle categorical variables (e.g., using Label Encoding or One-Hot Encoding)
categorical_columns = [
    "Status_checking", "Credit_history", "Purpose", "Savings_account", "Employment", 
    "Personal_status", "Other_debtors", "Property", "Other_installment_plans", 
    "Housing", "Job", "Telephone", "Foreign_worker"
]

# Label Encoding for categorical features
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features (X) and target (y)
X = df.drop(columns=["Class"])  # Drop the target column
y = df["Class"].apply(lambda x: 0 if x == 1 else 1)  # Class 1 is good, class 2 is bad (binary classification)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier (you can replace it with another classifier)
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate classification report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cost matrix based on the provided information (1 = Good, 2 = Bad)
# The cost matrix:
#    Predicted
#     1   2
#   -----------
# 1 | 0   1
# 2 | 5   0
cost_matrix = np.array([[0, 1], [5, 0]])

# Calculate the cost using the confusion matrix
total_cost = np.sum(cm * cost_matrix)
print(f"\nTotal Cost: {total_cost}")

