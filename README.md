# Quantum Reservoir Computing for Credit Risk Modeling


![Q1](https://github.com/user-attachments/assets/ff126db0-04df-45d6-a0be-c4cac7d13115)

## Overview

This project implements a **Quantum Reservoir Computing (QRC)** algorithm to classify individuals as either "Good" or "Bad" credit risks. The QRC leverages quantum computing principles to capture complex patterns in the data, offering a potential edge over classical methods in high-dimensional problems like credit risk modeling.

This document includes the project structure, methodology, code architecture, installation guide, and resources for further learning.

![Q2](https://github.com/user-attachments/assets/c4ce7253-f7f9-41ee-b2e5-a6fdb0f332fb)

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Code Architecture](#code-architecture)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Algorithm](#algorithm)
6. [Code Snippet](#code-snippet)
7. [Installation Guide](#installation-guide)
8. [References and Resources](#references-and-resources)
9. [Future Work](#future-work)

---

## Problem Statement

Credit risk modelling involves predicting whether a customer is a good or bad credit risk based on their financial and personal attributes. Misclassifying bad risks has significant financial implications. Our objective is to:
1. Improve recall for the "Bad" credit risk class above 0.5.
2. Implement QRC using quantum circuits to handle the dataset's complexity.

---

## Code Architecture

The code architecture consists of the following components:

1. **Data Preprocessing**:
    - Handles data cleaning, encoding, scaling, and train-test splitting.
2. **Quantum Reservoir**:
    - Constructs a quantum circuit reservoir to map input data to quantum states.
    - Utilizes quantum gates (Hadamard, CNOT, RX) to evolve the state.
3. **Readout Layer**:
    - A classical machine learning classifier that maps quantum outputs to labels.
4. **Evaluation Metrics**:
    - Calculates precision, recall, and F1-score to measure model performance.

---

## Project Structure

```
QuantumReservoirComputing/
├── data/
│   └── credit_data.csv        # Input dataset for credit risk modelling
├── src/
│   ├── preprocess.py          # Data preprocessing module
│   ├── quantum_reservoir.py   # Quantum reservoir implementation
│   ├── classifier.py          # Classical readout layer
│   ├── evaluation.py          # Model evaluation
│   └── main.py                # Entry point for running the project
├── README.md                  # Project documentation
└── requirements.txt           # Required Python packages
```

---

## Methodology

![Q3](https://github.com/user-attachments/assets/96a04659-a9a9-486a-a115-962dfdb82386)

1. **Data Preprocessing**:
    - Encode categorical features into numerical values.
    - Scale numerical features to a standard range.
    - Split the dataset into training and test sets.

2. **Quantum Reservoir Construction**:
    - Build a quantum circuit with several qubits representing the reservoir.
    - Apply quantum gates (Hadamard, RX, CNOT) to create a complex quantum state.
    - Encode classical data into quantum states.

3. **Classical Readout Layer**:
    - Use a classical classifier (SVM or logistic regression) to map quantum outputs to credit risk labels.

4. **Evaluation**:
    - Calculate recall, precision, and F1-score, focusing on improving recall for the "Bad" class.

---

## Algorithm

### Quantum Reservoir Computing (QRC) Algorithm:

1. **Input**: Credit risk data (`X`) and labels (`y`).
2. **Preprocessing**: Encode and scale the data.
3. **Quantum Reservoir Setup**:
    - Initialize a quantum circuit with `n_qubits`.
    - Encode data into quantum states using amplitude encoding.
    - Apply a series of quantum gates to evolve the quantum state.
4. **Feature Extraction**:
    - Measure the quantum circuit outputs.
    - Extract features from the quantum measurement probabilities.
5. **Classification**:
    - Train a classical classifier using the quantum features.
6. **Evaluation**:
    - Calculate recall, precision, and F1-score.

---

## Code Snippet

```python
from qiskit import QuantumCircuit, Aer, transpile
from sklearn.svm import SVC
from sklearn.metrics import recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Quantum Reservoir
def build_quantum_reservoir(n_qubits, data):
    qc = QuantumCircuit(n_qubits)
    for i, x in enumerate(data):
        qc.rx(x, i % n_qubits)  # Data encoding
    qc.h(range(n_qubits))  # Apply Hadamard gates
    qc.barrier()
    return qc

# Step 2: Feature Extraction
def extract_features(qc, n_shots=1024):
    simulator = Aer.get_backend('qasm_simulator')
    qc.measure_all()
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=n_shots).result()
    counts = result.get_counts()
    return [counts.get(f"{i:0{n_shots}b}", 0) / n_shots for i in range(2**qc.num_qubits)]

# Step 3: Classification
def classify(features, labels):
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(features, labels)
    return clf

# Example Workflow
data = np.random.rand(100, 2)  # Example dataset
scaled_data = StandardScaler().fit_transform(data)
quantum_circuits = [build_quantum_reservoir(3, x) for x in scaled_data]
features = np.array([extract_features(qc) for qc in quantum_circuits])
labels = np.random.randint(0, 2, 100)

# Train Classifier
classifier = classify(features, labels)
predictions = classifier.predict(features)
print(classification_report(labels, predictions))
```

---

## Installation Guide

### Prerequisites
1. Python 3.8 or higher
2. Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/QuantumReservoirComputing.git
   cd QuantumReservoirComputing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python src/main.py
   ```

---

## Final Result 

![Q6](https://github.com/user-attachments/assets/a59901de-c6fb-4352-ad33-a0358b24a8b2)

![Q7](https://github.com/user-attachments/assets/318b7456-5da4-4ca1-a030-11f3b9df2e1d)

![Q8](https://github.com/user-attachments/assets/2f54fe8e-7917-4598-89b2-2bcb014abc81)

## References and Resources

### Quantum SDKs
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [PennyLane Documentation](https://pennylane.ai/)

### Quantum Reservoir Computing
- *Paper*: Nakajima, K., *Reservoir Computing with Quantum Systems*.
- *Video*: [Quantum Machine Learning Introduction](https://www.youtube.com/watch?v=OV4YEJhOQ4A)

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Feature Scaling in ML](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## Future Work

- Experiment with other quantum encodings such as **density matrix representation**.
- Implement quantum-only readout layers for a fully quantum-based classification.
- Explore scalability for larger datasets and complex features.
- Investigate applications in other domains like healthcare and smart grids.

---

Feel free to contribute or suggest improvements to the project! 🚀


