import pandas as pd
import PreProcess as pp

'''x=pd.read_csv("train_dataset.csv")
mean=x['Age'].mean()
std=x['Age'].std()
print(mean,std)'''

x=pp.PreProcess("train_dataset.csv",17,.95)
z=pp.PreProcess("test_dataset.csv",17,.95)
y = pd.concat([x.getallData(), z.getallData()], ignore_index=True)
y.to_csv("All_features.csv", index=False)
