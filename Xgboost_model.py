import pandas as pd
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
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
 processedtrain_df = pd.read_csv("train processed_data.csv")
 #processedtrain_df.to_csv("train processed_data.csv", index=False)

 #print("\n=== Preprocessing Test Data ===")
 #pre_test = pp.PreProcess("test_dataset.csv", num_features=4,prun_factor=.95)  # Modify k (num_features) as needed
 processedtest_df = pd.read_csv("test processed_data.csv")
 #processedtest_df.to_csv("test processed_data.csv", index=False)

 # === 2. Split Features and Labels ===
 X_train = processedtrain_df.drop(columns='NObeyesdad')
 Y_train = processedtrain_df['NObeyesdad']
 X_test = processedtest_df.drop(columns='NObeyesdad')
 Y_test = processedtest_df['NObeyesdad']

 # === 3. XGBoost Model & GridSearch ===
 #print("\n=== Training Model with GridSearch ===")
 model = XGBClassifier(objective='multi:softprob', num_class=len(Y_train.unique()), eval_metric='mlogloss', random_state=42,learning_rate=0.1, max_depth=5, n_estimators=150, subsample=0.8, colsample_bytree=1.0)
 model.fit(X_train, Y_train)
 #param_grid = {
 #   'learning_rate': [0.01, 0.1, 0.2],
#    'max_depth': [3, 5, 7],
 #   'n_estimators': [50, 100, 150],
 #   'subsample': [0.8, 1.0],
 #   'colsample_bytree': [0.8, 1.0]
 # }

 #grid = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
 #grid.fit(X_train, Y_train)
 best_model =model
 #print("Best Parameters:", grid.best_params_)

 # === 4. Evaluate with Cross-Validation ===
 #print("\n=== Cross-Validation Results ===")
 #scores = cross_val_score(best_model, X_train, Y_train, cv=5)
 #print(f"Cross-validation accuracy scores: {scores}")
 #print(f"Mean CV Accuracy: {scores.mean():.4f}")

  #=== 5. Train vs Test Accuracy to Check Overfitting ===
 #print("\n=== Overfitting Check ===")
 #train_preds = best_model.predict(X_train)
 #test_preds = best_model.predict(X_test)
 #train_acc = accuracy_score(Y_train, train_preds)
 #test_acc = accuracy_score(Y_test, test_preds)
 #print(f"Train Accuracy: {train_acc:.4f}")
 #print(f"Test Accuracy: {test_acc:.4f}")
 #if train_acc - test_acc < 0.15:
    #print("There is not overfitting :)")
 #else:
    #print("Unfortunately, there is overfitting :(")
 '''
 # === 6. Final Evaluation ===

 print("\n=== Classification Report ===")
 print("\nClassification Report:\n", classification_report(Y_test, test_preds, target_names=pre_train.labels))

 # === 7. Confusion Matrix ===
 print("\n=== Confusion Matrix ===")

 disp = ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, Y_test,
    display_labels=pre_train.labels,
    cmap='Blues', xticks_rotation='vertical'
 )

 plt.title("Confusion Matrix")
 disp.plot()
 plt.show() '''
 return predict(standardize(raw_data), best_model)
load_model( {
    'FCVC': 1,
    'Meal_Regularity': 1,
    'Weight': 60,
    'Height': 1.60,
    'Gender': 1,
    'CaloricIntake': 1800,
    'TUE': 2,
    'Age': 25
})