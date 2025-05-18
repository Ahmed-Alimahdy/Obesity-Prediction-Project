import PreProcess as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
 print("=== Preprocessing Training Data ===")
 #pre_train = pp.PreProcess("train_dataset.csv", num_features=4,prun_factor=.95)  # Modify k (num_features) as needed
 processedtrain_df = pd.read_csv("train processed_data.csv")
 #processedtrain_df.to_csv("train processed_data.csv", index=False)

 print("\n=== Preprocessing Test Data ===")
 #pre_test = pp.PreProcess("test_dataset.csv", num_features=4,prun_factor=.95)  # Modify k (num_features) as needed
 processedtest_df = pd.read_csv("test processed_data.csv")
 #processedtest_df.to_csv("test processed_data.csv", index=False)

 # === 2. Split Features and Labels ===
 X_train=processedtrain_df.drop(columns='NObeyesdad')
 Y_train=processedtrain_df['NObeyesdad']
 X_test=processedtest_df.drop(columns='NObeyesdad')
 Y_test=processedtest_df['NObeyesdad']

   



 # === 3. KNN models & GridSearch ===
 #print("Training KNN models...")
 # Find the optimal K value
 #k_values = list(range(1, 21))
 #accuracy_scores = []
 #for k in k_values:
  #     knn = KNeighborsClassifier(n_neighbors=k)
 #       knn.fit(X_train, Y_train)
   #     y_pred = knn.predict(X_test)
  #      accuracy = accuracy_score(Y_test, y_pred)
   #     accuracy_scores.append(accuracy)
   #     print(f"K={k}, Accuracy: {accuracy:.4f}")
 # Find the best K value
 #best_k = k_values[np.argmax(accuracy_scores)]
 #print(f"\nBest K value: {best_k} with accuracy: {max(accuracy_scores):.4f}")

 # Train the final model with K from ^above^
 final_knn = KNeighborsClassifier(n_neighbors=1)
 final_knn.fit(X_train, Y_train)

 # Evaluate the model (console overview,to be replaced with gui)
 y_pred = final_knn.predict(X_test)

 # === 5. Train vs Test Accuracy to Check Overfitting ===
 #test_acc = accuracy_score(Y_test, y_pred)
 #print(f"Train Accuracy: {test_acc:.4f}")
 #print(f"Test Accuracy: {accuracy:.4f}")
 #if(accuracy-test_acc < .15):
  #print("there is not overfitting :)")
 #else:
 # print("Unfortunatly, there is overfitting :(")
 #print("\nClassification Report:")
 #print(classification_report(Y_test, y_pred))

 #print("\nConfusion Matrix:")
 cm = confusion_matrix(Y_test, y_pred)
 # print(cm)

 # Visualize results of Plot accuracy vs k value (not needed,gui potential)
 #plt.figure(figsize=(10, 6))
 #plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
 #plt.title('Accuracy vs. K Value')
 #plt.xlabel('K Value')
 #plt.ylabel('Accuracy')
 #plt.grid(True)
 #plt.savefig('knn_accuracy_vs_k.png')

 # Plot confusion matrix (to be replaced with gui)
 #plt.figure(figsize=(10, 8))
 #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
 #plt.title('Confusion Matrix')
 #plt.xlabel('Predicted Label')
 #plt.ylabel('True Label')
 #plt.savefig('knn_confusion_matrix.png')

 #print("Complete. Check the generated files.")
 return predict(standardize(raw_data), final_knn)

