from operator import le
import pandas as pd
from sklearn import svm
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
 pre_train = pp.PreProcess("train processed_data.csv")  # Modify k (num_features) as needed
 processedtrain_df = pd.read_csv("train processed_data.csv")
 #processedtrain_df.to_csv("train processed_data.csv", index=False)

 #print("\n=== Preprocessing Test Data ===")
 pre_test = pp.PreProcess("test processed_data.csv")  # Modify k (num_features) as needed
 processedtest_df = pd.read_csv("test processed_data.csv")
 #processedtest_df.to_csv("test processed_data.csv", index=False)

 # === 2. Split Features and Labels ===
 X_train=processedtrain_df.drop(columns='NObeyesdad')
 Y_train=processedtrain_df['NObeyesdad']
 X_test=processedtest_df.drop(columns='NObeyesdad')
 Y_test=processedtest_df['NObeyesdad']

 # === 3. SVM Model & GridSearch ===
 #print("\n=== Training Model with GridSearch ===")
 model=svm.SVC(kernel= 'rbf',C=100,gamma='scale', decision_function_shape='ovo',class_weight='balanced',random_state=42)
 model.fit(X_train, Y_train)
 #param_grid = {
 #   'C': [0.1, 1, 10, 100],
  #   'gamma': ['scale', 'auto', 0.1, 0.01],
 #    'decision_function_shape': ['ovo', 'ovr']
# }
 #grid = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
 #grid.fit(X_train, Y_train)
 #best_model = grid.best_estimator_  
 #print("Best Parameters:", grid.best_params_)

 # === 4. Evaluate with Cross-Validation ===
 #print("\n=== Cross-Validation Results ===")
 #scores = cross_val_score(model, X_train, Y_train, cv=5)
 #print(f"Cross-validation accuracy scores: {scores}")
 #print(f"Mean CV Accuracy: {scores.mean():.4f}")

 # === 5. Train vs Test Accuracy to Check Overfitting ===
 #print("\n=== Overfitting Check ===")
 #train_preds = model.predict(X_train)
 #test_preds = model.predict(X_test)
 #train_acc = accuracy_score(Y_train, train_preds)
 #test_acc = accuracy_score(Y_test, test_preds)
 #print(f"Train Accuracy: {train_acc:.4f}")
 #print(f"Test Accuracy: {test_acc:.4f}")
 #if(train_acc-test_acc < .15):
 # print("there is not overfitting :)")
 #else:
  #print("Unfortunatly, there is overfitting :(")
 return predict(standardize(raw_data), model)




    



