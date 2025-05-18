
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV,cross_val_score
import PreProcess as pp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
def predict(new_data, model):
    # Convert to DataFrame
    new_df = pd.DataFrame([new_data])
    
    labels = [
    'Insufficient_Weight', 
    'Normal_Weight', 
    'Obesity_Type_I', 
    'Obesity_Type_II', 
    'Obesity_Type_III', 
    'Overweight_Level_I', 
    'Overweight_Level_II'
    ]
    # Get prediction
    prediction = model.predict(new_df)
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
def load_model(raw_data):
 # === 1. Preprocessing ===
 #print("=== Preprocessing Training Data ===")
 #pre_train = pp.PreProcess("train_dataset.csv")  # Modify k (num_features) as needed
 #processedtrain_df = pre_train.getselectiondata()
 processedtrain_df=pd.read_csv("train processed_data.csv")

 #print("\n=== Preprocessing Test Data ===")
 #pre_test = pp.PreProcess("test_dataset.csv", num_features=3,prun_factor=.95)  # Modify k (num_features) as needed
 processedtest_df=pd.read_csv("test processed_data.csv")
 # === 2. Split Features and Labels ===
 X_train=processedtrain_df.drop(columns='NObeyesdad')
 Y_train=processedtrain_df['NObeyesdad']
 X_test=processedtest_df.drop(columns='NObeyesdad')
 Y_test=processedtest_df['NObeyesdad']

 # === 3. Naive Bayes Model ===
 #print("\n=== Training Naive Bayes Model ===")

 nb_model = GaussianNB(var_smoothing=.01)
 params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
 grid = GridSearchCV(GaussianNB(), params, cv=5)
 grid.fit(X_train, Y_train)
 #print("Best var_smoothing:", grid.best_params_)
 nb_model = grid.best_estimator_
 # === 4. Evaluate with Cross-Validation ===
 #print("\n=== Cross-Validation Results ===")
 #nb_scores = cross_val_score(nb_model, X_train, Y_train, cv=5)
 #print(f"Cross-validation accuracy scores: {nb_scores}")
 #print(f"Mean CV Accuracy: {nb_scores.mean():.4f}")

 # === 5. Train vs Test Accuracy to Check Overfitting ===
 #print("\n=== Overfitting Check ===")
 #nb_train_preds = nb_model.predict(X_train)
 #nb_test_preds = nb_model.predict(X_test)
 #nb_train_acc = accuracy_score(Y_train, nb_train_preds)
 #nb_test_acc = accuracy_score(Y_test, nb_test_preds)
 #print(f"Train Accuracy: {nb_train_acc:.4f}")
 #print(f"Test Accuracy: {nb_test_acc:.4f}")
 #if nb_train_acc - nb_test_acc < 0.15:
 #   print("There is not overfitting :)")
 #else:
  #  print("Unfortunately, there is overfitting :(")
 #return predict(standardize(raw_data), nb_model)

'''
# === 6. Final Evaluation ===
print("\n=== Classification Report ===")
print("\nClassification Report:\n", classification_report(Y_test, nb_test_preds, target_names=pre_train.labels))

# === 7. Confusion Matrix ===
print("\n=== Confusion Matrix ===")
nb_disp = ConfusionMatrixDisplay.from_estimator(
    nb_model, X_test, Y_test,
    display_labels=pre_train.labels,
    cmap='Blues', xticks_rotation='vertical'
)
plt.title("Confusion Matrix - Naive Bayes")
nb_disp.plot()
plt.show()
'''