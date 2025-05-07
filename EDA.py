import seaborn as sns
import matplotlib.pyplot as plt
class EDA:
    def __init__(self, df):
        self.df = df
    def showHistogram(self, column):
        sns.histplot(self.df[column], bins=30, kde=True)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def showBoxPlot(self, column):
        sns.boxplot(x=self.df[column],color='red')
        plt.title(f"Box Plot of {column}")
        plt.xlabel(column)
        plt.show()

    def showscatterPlot(self, x_column, y_column):
        sns.scatterplot(x=self.df[x_column], y=self.df[y_column],color='green')
        plt.title(f"Scatter Plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def showCorrelation(self):
        plt.figure(figsize=(14, 5))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm',  cbar=True, linewidths=0.1)
        plt.title("Correlation Matrix")
        plt.show()

    def showstatistics(self,column):
       return self.df[column].describe().round(3)
