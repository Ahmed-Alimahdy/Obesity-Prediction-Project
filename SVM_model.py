from operator import le
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV,cross_val_score
import PreProcess as pp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# === 1. Preprocessing ===
print("=== Preprocessing Training Data ===")
pre_train = pp.PreProcess("train_dataset.csv", num_features=12)  # Modify k (num_features) as needed
processedtrain_df = pre_train.getselectiondata()
processedtrain_df.to_csv("train processed_data.csv", index=False)

print("\n=== Preprocessing Test Data ===")
pre_test = pp.PreProcess("test_dataset.csv", num_features=12)  # Modify k (num_features) as needed
processedtest_df = pre_test.getallData()[processedtrain_df.columns]
processedtest_df.to_csv("test processed_data.csv", index=False)

# === 2. Split Features and Labels ===
X_train=processedtrain_df.drop(columns='NObeyesdad')
Y_train=processedtrain_df['NObeyesdad']
X_test=processedtest_df.drop(columns='NObeyesdad')
Y_test=processedtest_df['NObeyesdad']

# === 3. SVM Model & GridSearch ===
print("\n=== Training Model with GridSearch ===")
model=svm.SVC(kernel= 'rbf',C=1,gamma='scale', decision_function_shape='ovo',class_weight='balanced',random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'decision_function_shape': ['ovo', 'ovr']
}
grid = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
grid.fit(X_train, Y_train)
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# === 4. Evaluate with Cross-Validation ===
print("\n=== Cross-Validation Results ===")
scores = cross_val_score(best_model, X_train, Y_train, cv=5)
print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean():.4f}")

# === 5. Train vs Test Accuracy to Check Overfitting ===
print("\n=== Overfitting Check ===")
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)
train_acc = accuracy_score(Y_train, train_preds)
test_acc = accuracy_score(Y_test, test_preds)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
if(train_acc-test_acc < .02):
 print("there is not overfitting :)")
else:
 print("Unfortunatly, there is overfitting :(")

# === 6. Final Evaluation ===
print("\n=== Classification Report ===")
print("\nClassification Report:\n", classification_report(Y_test, test_preds, target_names=pre_train.label_encoders['NObeyesdad'].classes_))

# === 7. Confusion Matrix ===
print("\n=== Confusion Matrix ===")
disp = ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, Y_test,
    display_labels=pre_train.label_encoders['NObeyesdad'].classes_,
    cmap='Blues', xticks_rotation='vertical'
)
plt.title("Confusion Matrix")
disp.plot()
plt.show()






