import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import EDA
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif,RFE
from sklearn.preprocessing import PowerTransformer, LabelEncoder,StandardScaler,MinMaxScaler

class PreProcess:
    def __init__(self, data_csv, num_features,prun_factor):
        self.df = pd.read_csv(data_csv)
        self.__handleDublicates()
        self.__handleNulls()  
        #self.__convertData(self.__getCategorical())
        numerics = self.__getNumerics()  
        categoricals = self.__getCategorical()
        self.__encodeCategorical(self.__getCategorical())
        self.df=self.feature_engineering()  
        self.__handleOutliers(numerics)
        # change one of the features scaling according to the model
        #self.__powerTransform(numerics)
        self.__Standardization(numerics)
        #self.__normalization(numerics)
        self.Correlation_Pruning()
        #self.selected_columns = self.__featureSelection(numerics, categoricals, num_features)
        self.topFeatures = self.HybridFeatureSelection(num_features)

    def __handleNulls(self):
        self.df.fillna({'CALC': 'Unknown', 'FCVC': self.df['FCVC'].mean()}, inplace=True)

    def __handleDublicates(self):
        print(f"Sum of duplicated data is {self.df.duplicated().sum()} is dropped")
        self.df.drop_duplicates()

    def __getCategorical(self):
     # Check for columns with non-numeric data types
     categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
     print("Detected categorical columns:", categorical_columns)  # Debugging
     return categorical_columns

    def __getNumerics(self):
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def __convertData(self, catData):
        for obj in catData:
            self.df[obj] = self.df[obj].astype('category')
            print(self.df[obj])
            
   

    def HybridFeatureSelection(self, num_features):
     X = self.df.drop(columns=['NObeyesdad'])
     y = LabelEncoder().fit_transform(self.df['NObeyesdad'])
    
     # 1. Mutual Information (existing)
     mi_scores = mutual_info_classif(X, y)
     mi_features = X.columns[np.argsort(mi_scores)[-num_features:]]
     print("Mutual Information: ",mi_features)

     # 2. RFE with Logistic Regression
     model = LogisticRegression(max_iter=1000)
     rfe = RFE(model, n_features_to_select=num_features)
     rfe.fit(X, y)
     rfe_features = X.columns[rfe.support_]
     print("LogisticRegression: ",rfe_features)
    
     # 3. XGBoost Importance
     xgb = XGBClassifier()
     xgb.fit(X, y)
     xgb_features = X.columns[np.argsort(xgb.feature_importances_)[-num_features:]]
     print("XGBoost: ",xgb_features)
     
     # Get consensus features
     all_features = (set(mi_features) | set(rfe_features)) | set(xgb_features)
     print("all features: ",all_features)
     return list(all_features)

    def __encodeCategorical(self, catData):
     # Store label encoders for potential inverse transform
     self.label_encoders = {}  
     for col in catData:
        # Initialize and fit label encoder
        le = LabelEncoder()
        self.df[col] = le.fit_transform(self.df[col])
        self.label_encoders[col] = le  # Store encoder
        # Debug info
        print(f"Encoded column '{col}' with {len(le.classes_)} unique classes.")
        print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        print("Categorical data encoding completed.")
    
    def __handleOutliers(self, numData):
        print("Handling outliers for columns:", numData)
        for num in numData:
            Q1 = self.df[num].quantile(0.25)
            Q3 = self.df[num].quantile(0.75)
            IQR = Q3 - Q1
            lowerBound = Q1 - 1.5 * IQR
            upperBound = Q3 + 1.5 * IQR
            self.df[num] = self.df[num].clip(lower=lowerBound, upper=upperBound)

    def __powerTransform(self, numData):
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        self.df[numData] = transformer.fit_transform(self.df[numData])
        print("Applying power transformation to:", numData)

    def __Standardization(self,numData):
        scaler=StandardScaler()
        self.df[numData]=scaler.fit_transform(self.df[numData])
        print("Applying Standardization to:", numData)
    
    def __normalization(self,numData):
        scaler=MinMaxScaler()
        self.df[numData]=scaler.fit_transform(self.df[numData])
        print("Applying normalization to:", numData)
        
    def getallData(self):
     """Reorder columns to place the target column 'NObeyesdad' as the last column."""
     if 'NObeyesdad' in self.df.columns:
        # Move 'NObeyesdad' to the last column
        columns = [col for col in self.df.columns if col != 'NObeyesdad'] + ['NObeyesdad']
        self.df = self.df[columns]
     return self.df.copy()
    
    def getselectiondata(self):
       features = self.topFeatures  # Make a copy to avoid modifying the original
       if('NObeyesdad' not in features):
        features.append('NObeyesdad')
       return self.df[features].copy()
    
    def feature_engineering(self):
     self.df['Height_m'] = self.df['Height'] / 100
     self.df['BMI'] = self.df['Weight'] / (self.df['Height_m'] ** 2)
     self.df['BMI_Prime'] = self.df['BMI'] / 25
     self.df['Metabolic_Age'] = self.df['Age'] * (1 + (self.df['BMI']  / 25))
     self.df['WHtR'] = self.df['Weight'] / self.df['Height']
     self.df['Age_squared'] = self.df['Age'] ** 2
     self.df['CaloricIntake'] = self.df['CH2O'] + self.df['FAF'] + self.df['FCVC']
     self.df['family_risk'] = self.df['family_history_with_overweight'] * (self.df['Age'] / 30.0)
     self.df['sedentary_score'] = (5 - self.df['FAF']) * self.df['TUE']
     self.df['hydration_score'] = self.df['CH2O'] * (self.df['FCVC'] / 3.0)
     self.df['Unhealthy_Score'] = (self.df['SMOKE'] + self.df['CALC'] + self.df['FAVC']) / 3.0
     self.df['Activity_Hydration'] = self.df['CH2O'] * (self.df['FAF'] / 3.0)
     self.df['Meal_Regularity'] = self.df['NCP'].apply(lambda x: 1 if x == 3 else 0.5 if x == 2 else 0)
     return self.df
    
    # to prevent the redundancy of features 
    def Correlation_Pruning(self):
       corr_matrix = self.df.corr().abs()
       upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
       to_drop = [col for col in upper.columns if any(upper[col] > self.prun_factor)]
       self.df = self.df.drop(columns=to_drop)
    
#Read  the dataset and process it




