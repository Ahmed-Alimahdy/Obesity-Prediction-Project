import pandas as pd

class PreProcess:
    def __init__(self, data_csv):
        self.df = pd.read_csv(data_csv)
        self.handleNulls()
        self.convertData(self.getCategorical())
        self.encodeCategorical(self.getCategorical())
        self.normalize()
        
    # Handle Null Values
    def handleNulls(self):
        self.df.fillna({'CALC':  'Unknown', 'FCVC': self.df['FCVC'].mean()}, inplace=True)
        # print(df.isna().sum())
        
    # Extracts Categorical Data
    def getCategorical(self):
        return ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
        
    
    # Convert Data to Suitable Data Types
    def convertData(self, catData):
        ojbects = catData
        for obj in ojbects:
            self.df[obj] = self.df[obj].astype('category')
        # print(self.df.info())

    # Encode Categorical Data
    def encodeCategorical(self, catData):
        self.df = pd.get_dummies(self.df, columns=catData)
        # print(self.df.info())
    
    # Normalize Numerical Values
    def normalize(self):
        numerics = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for num in numerics:
            self.df[num] = ((self.df[num] - self.df[num].min()) / (self.df[num].max() - self.df[num].min()))

preProc = PreProcess("train_dataset.csv")