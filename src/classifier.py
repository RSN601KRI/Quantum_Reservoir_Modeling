from sklearn.linear_model import LogisticRegression

def train_classifier(features, labels):
    classifier = LogisticRegression(random_state=42)
    classifier.fit(features, labels)
    return classifier
