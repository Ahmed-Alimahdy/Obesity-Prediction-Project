import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif

class PreProcess:
    def __init__(self, data_csv):
        self.df = pd.read_csv(data_csv)
        self.__handleNulls()
        self.__convertData(self.__getCategorical())
        self.__encodeCategorical(self.__getCategorical())
        self.__handleOutliers(self.__getNumerics())
        self.__normalize(self.__getNumerics())

    def __handleNulls(self):
        self.df.fillna({'CALC':  'Unknown', 'FCVC': self.df['FCVC'].mean()}, inplace=True)
        
    # Extract Categorical Data
    def __getCategorical(self):
        return self.df.select_dtypes(include=['object']).columns.tolist()
    
    # Extract Numerical data
    def __getNumerics(self):
        return self.df.select_dtypes(include=['number']).columns.tolist()
        
    # Convert Data to Suitable Data Types
    def __convertData(self, catData):
        for obj in catData:
            self.df[obj] = self.df[obj].astype('category')
        
    def __encodeCategorical(self, catData):
        self.df = pd.get_dummies(self.df, columns=catData)
    
    def __handleOutliers(self, numData):
        num_df = self.df[numData]
        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5*IQR
        upperBound = Q3 + 1.5*IQR
        
        for num in numData:
            self.df[num] = self.df[num].clip(lower=lowerBound, upper=upperBound)

    def __normalize(self, numData):
        for num in numData:
            col = self.df[num]
            self.df[num] = ((col - col.min()) / (col.max() - col.min()))
            

# Test
# preProc = PreProcess("train_dataset.csv")