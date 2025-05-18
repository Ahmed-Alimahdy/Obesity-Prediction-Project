from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import PreProcess as pp
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

 # === 3. Random forest Model ===
 rf_classifier = RandomForestClassifier(random_state=42)
 rf_classifier.fit(X_train, Y_train)

 # === 4. Train vs Test Accuracy to Check Overfitting ===
 #print("\n=== Overfitting Check ===")
 #train_preds = rf_classifier.predict(X_train)
 #test_preds = rf_classifier.predict(X_test)
 #train_acc = accuracy_score(Y_train, train_preds)
 #test_acc = accuracy_score(Y_test, test_preds)
 #print(f"Train Accuracy: {train_acc:.4f}")
 #print(f"Test Accuracy: {test_acc:.4f}")
 #if train_acc - test_acc < 0.15:
    #print("There is not overfitting :)")
 #else:
    #print("Unfortunately, there is overfitting :(")
 return predict(standardize(raw_data), rf_classifier)
'''
# === 6. Final Evaluation ===

print("\n=== Classification Report ===")
print("\nClassification Report:\n", classification_report(Y_test, test_preds, target_names=pre_train.labels))

# === 7. Confusion Matrix ===
print("\n=== Confusion Matrix ===")

disp = ConfusionMatrixDisplay.from_estimator(
    rf_classifier, X_test, Y_test,
    display_labels=pre_train.labels,
    cmap='Blues', xticks_rotation='vertical'
)

plt.title("Confusion Matrix")
disp.plot()
plt.show()
'''
