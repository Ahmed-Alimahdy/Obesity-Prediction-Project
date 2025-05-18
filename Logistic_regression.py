import numpy as np
import pandas as pd
import PreProcess as pp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
weights_nums=[[ 3.28532402e-02, -5.47011780e-01, -1.86224777e+00, -1.40611340e+00,
   6.15364735e+00, -8.36871052e-01, -1.53425659e+00],
 [-2.91204539e+00, -1.40166559e+00, -6.26654770e-01, -1.36592963e+00,
   1.10651035e+01, -3.09210290e+00, -1.66670525e+00],
 [-1.18452244e+02, -8.43996363e+01,  6.00711012e+01,  8.11772833e+01,
   7.89406540e+01, -3.68070764e+01,  1.94699184e+01],
 [ 3.34683723e+01,  2.55648492e+01, -1.90726580e+01, -2.60624467e+01,
  -1.98089582e+01,  1.17196819e+01, -5.80884051e+00],
 [ 3.42837169e+00,  4.54548625e+00, -7.51650075e-01,  4.34947761e+00,
  -1.68383778e+01,  2.45129639e+00,  2.81539599e+00],
 [ 4.56182863e-02,  9.07424475e-02,  4.76223979e-01, -1.04862184e-01,
  -5.48908776e-01, -1.62680547e-01,  2.03866795e-01],
 [-3.10151638e-01, -3.41910972e-01,  5.09494984e-01, -3.76021864e-01,
  -1.54133495e-01,  1.07951765e-02,  6.61927810e-01],
 [-6.16341231e-01, -4.30493117e-01,  4.06220622e-01,  1.86838413e+00,
  -1.25080540e+00, -1.50807516e-01,  1.73842510e-01],
 [-2.97280152e+01, -9.52152720e+00,  1.23828880e+01,  5.13516305e+00,
   3.39388122e+00,  5.42402975e+00,  1.29135804e+01]]
biases_nums=[[-29.72801522,  -9.5215272,   12.382888 ,    5.13516305 ,  3.39388122,
    5.42402975 , 12.91358039]]
# === 1. Softmax Function ===
def predict_new(new_data, model):
    # Convert to DataFrame
    new_df = pd.DataFrame([new_data])
    weights, biases = model

    # Compute logits and probabilities
    # Add bias term to new data
    new_X = np.hstack([new_df.values, np.ones((new_df.shape[0], 1))])
    logits = np.dot(new_X, weights) + biases
    probs = softmax(logits)
    prediction = np.argmax(probs, axis=1)

    labels = [
        'Insufficient_Weight', 
        'Normal_Weight', 
        'Obesity_Type_I', 
        'Obesity_Type_II', 
        'Obesity_Type_III', 
        'Overweight_Level_I', 
        'Overweight_Level_II'
    ]
    return labels[prediction[0]]
       

def standardize(raw_data):
   standardized_data = {}
   x=pd.read_csv("All_features.csv")
   for key, value in raw_data.items():
     mean = x[key].mean()
     std = x[key].std()
     if std != 0:
        standardized_value = (value - mean) / std
     else:
        standardized_value = 0 
    # print(key , "mean: ",mean,"std: ",std,"standerdize: ",standardized_value)
     standardized_data[key] = float(standardized_value)
   return standardized_data
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

def train_softmax(X, y, num_classes, lr=2.5, epochs=15000):
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
def load_model(raw_data):
 print("=== Preprocessing Training Data ===")
 #pre_train = pp.PreProcess("train_dataset.csv")
 #processedtrain_df = pre_train.getselectiondata()
 processedtrain_df=pd.read_csv("train processed_data.csv")

 print("\n=== Preprocessing Test Data ===")
 #pre_test = pp.PreProcess("test_dataset.csv", num_features=5,prun_factor=.8)
 #processedtest_df = pre_test.getallData()[processedtrain_df.columns]
 processedtest_df=pd.read_csv("test processed_data.csv")

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
 #weights, biases = train_softmax(X_train_bias, Y_train_oh, 7)
 weights= weights_nums
 biases= biases_nums
 # === 9. Evaluate ===
 #train_preds = predict(X_train_bias, weights, biases)
 #test_preds = predict(X_test_bias, weights, biases)

 #print("\nTrain Accuracy:", accuracy_score(Y_train, train_preds))
 #print("Test Accuracy:", accuracy_score(Y_test, test_preds))
 #print("\nConfusion Matrix:\n", confusion_matrix(Y_test, test_preds))
 #print(weights, biases)
 return predict_new(standardize(raw_data), (weights, biases))