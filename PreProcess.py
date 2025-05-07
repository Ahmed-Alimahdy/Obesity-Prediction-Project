import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import PowerTransformer
class PreProcess:
    def __init__(self, data_csv):
        self.df = pd.read_csv(data_csv)
        self.__handleNulls()
        self.__convertData(self.__getCategorical())
        self.__encodeCategorical(self.__getCategorical())
        numerics = self.__getNumerics()  # Refresh after encoding
        self.__handleOutliers(numerics)
        #self.__normalize(numerics)
        self.__powerTransform(numerics)
        self.selected_columns = self.__featureSelection(numerics, k=5)
    def __handleNulls(self):
        if 'FCVC' in self.df.columns:
            self.df.fillna({'CALC': 'Unknown', 'FCVC': self.df['FCVC'].mean()}, inplace=True)
        else:
            print("Warning: 'FCVC' column not found in the dataset. Skipping mean imputation for 'FCVC'.")
            self.df.fillna({'CALC': 'Unknown'}, inplace=True)

    def __getCategorical(self):
        return self.df.select_dtypes(include=['object']).columns.tolist()

    def __getNumerics(self):
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def __convertData(self, catData):
        for obj in catData:
            self.df[obj] = self.df[obj].astype('category')

    def __encodeCategorical(self, catData):
        self.df = pd.get_dummies(self.df, columns=catData)

    def __handleOutliers(self, numData):
        print("Handling outliers for columns:", numData) 
        for num in numData:
            Q1 = self.df[num].quantile(0.25)
            Q3 = self.df[num].quantile(0.75)
            IQR = Q3 - Q1
            lowerBound = Q1 - 1.5 * IQR
            upperBound = Q3 + 1.5 * IQR
            self.df[num] = self.df[num].clip(lower=lowerBound, upper=upperBound)

    def __normalize(self, numData):
        print("Normalizing columns:", numData)
        for num in numData:
            col = self.df[num]
            range_val = col.max() - col.min()
            if range_val != 0:
                self.df[num] = (col - col.min()) / range_val
            else:
                self.df[num] = 0
    # second form of normalization handle the outliers & skewness
    def __powerTransform(self, numData):
     print("Applying power transformation to:", numData)
     transformer = PowerTransformer(method='yeo-johnson', standardize=False)
     self.df[numData] = transformer.fit_transform(self.df[numData])
    def __featureSelection(self, numerics, k):
        # SelectKBest for feature selection using ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=k)
        X_numerics = self.df[numerics]  # Only use numerical features for selection
        selector.fit(X_numerics, self.df['NObeyesdad'])  # Assuming 'target' is your label column
        
        # Get the selected feature names
        selected_columns = X_numerics.columns[selector.get_support()]
        print("Selected features:", selected_columns)
        return selected_columns

    def getData(self):
        # Return a DataFrame with only selected features
        return self.df[list(self.selected_columns)].copy()


pre = PreProcess("train_dataset.csv")
processed_df = pre.getData()
processed_df.to_csv("processed_data.csv", index=False)
