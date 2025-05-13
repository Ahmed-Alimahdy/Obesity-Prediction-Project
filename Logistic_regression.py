import numpy as np
import pandas as pd
import PreProcess as pp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# === 1. Softmax Function ===
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# === 2. Cross-Entropy Loss with Class Weights ===
def cross_entropy(y_true, y_pred, class_weights=None):
    loss = -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)
    if class_weights is not None:
        weights = np.array([class_weights[np.argmax(y)] for y in y_true])
        loss = loss * weights
    return np.mean(loss)

def train_softmax(X, y, num_classes, lr=2.5, epochs=20000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, num_classes))  # Fix: Remove +1 since bias is in X
    biases = np.zeros((1, num_classes))

    # Compute class weights (inverse frequency)
    class_counts = np.sum(y, axis=0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes

    prev_loss = float('inf')
    for epoch in range(epochs):
        logits = np.dot(X, weights) + biases
        probs = softmax(logits)
        loss = cross_entropy(y, probs, class_weights)
        if abs(prev_loss - loss) < 1e-6:  # Convergence check
            print(f"Converged at epoch {epoch}, Loss: {loss:.4f}")
            break
        grad_w = np.dot(X.T, (probs - y)) / n_samples
        grad_b = np.sum(probs - y, axis=0, keepdims=True) / n_samples
        weights -= lr * grad_w
        biases -= lr * grad_b
        prev_loss = loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, biases
# === 7. Prediction Function ===
def predict(X, weights, biases):
    logits = np.dot(X, weights) + biases
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

# === 3. One-Hot Encoding ===
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]
# === 2. Preprocessing ===
print("=== Preprocessing Training Data ===")
pre_train = pp.PreProcess("train_dataset.csv", num_features=12,prun_factor=.8)
processedtrain_df = pre_train.getselectiondata()
processedtrain_df.to_csv("train processed_data.csv", index=False)

print("\n=== Preprocessing Test Data ===")
pre_test = pp.PreProcess("test_dataset.csv", num_features=12,prun_factor=.8)
processedtest_df = pre_test.getallData()[processedtrain_df.columns]
processedtest_df.to_csv("test processed_data.csv", index=False)

# === 2. Split Features and Labels ===
X_train = processedtrain_df.drop(columns='NObeyesdad')
Y_train = processedtrain_df['NObeyesdad']
X_test = processedtest_df.drop(columns='NObeyesdad')
Y_test = processedtest_df['NObeyesdad']

# Encode labels consistently
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)

# Add bias term
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# === 3. One-Hot Encoding ===
Y_train_oh = one_hot(Y_train, 7)

# === 8. Train the Model ===
weights, biases = train_softmax(X_train_bias, Y_train_oh, 7)

# === 9. Evaluate ===
train_preds = predict(X_train_bias, weights, biases)
test_preds = predict(X_test_bias, weights, biases)

print("\nTrain Accuracy:", accuracy_score(Y_train, train_preds))
print("Test Accuracy:", accuracy_score(Y_test, test_preds))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, test_preds))
#print("\nClassification Report:\n", classification_report(Y_test, test_preds, target_names=label_encoder.classes_))