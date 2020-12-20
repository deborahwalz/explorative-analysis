import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class Analysis():
    """
    Class for first explorative data analysis.
    """

    def __init__(self, data):
        self.data = data
        self.columns = data.columns


    def columns_with_na(self, printing=False):
        """
        Select columns which contain missing values.
        If printing=True, print the percentage of missing values for each column with missing values.
        """
        self.cols_with_na = [var for var in self.columns if self.data[var].isnull().sum()>1]

        if printing:
            print("Percentage of missing values:")
            print()
            print(self.data[self.cols_with_na].isnull().mean())


    def num_columns_with_na(self, printing=False):
        """
        Select numerical columns which contain missing values.
        If printing=True, print the percentage of missing values for each column with missing values.
        """
        self.num_cols_with_na = [var for var in self.columns if self.data[var].isnull().sum()>1]

        if printing:
            print("Percentage of missing values:")
            print()
            print(self.data[self.num_cols_with_na].isnull().mean())
            

    def plot_compare_na(self, col_na, col_target):
        """
        Group the column col_na with 1/0 and plot the median of col_target for each group.
        """
        data = self.data.copy()

        # 1 if missing value, 0 otherwise
        data[col_na] = np.where(data[col_na].isnull(), 1, 0)
    
        data.groupby(col_na)[col_target].median().plot.bar()
        plt.title(col_na)
        plt.show()
        
