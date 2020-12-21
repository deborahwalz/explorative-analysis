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


    def select_cols_types(self, exclude=None):
        """
        Select all categorical, numerical and discrete columns.
        Exclude: Excluding discrete columns (e.g. "ID")
        """
        self.categorical_cols = [var for var in self.columns if self.data[var].dtypes=="O"]
        self.num_cols = [var for var in self.columns if self.data[var].dtypes!="O"]
        self.discrete_cols = [var for var in self.num_cols if (self.data[var].nunique()<20 
                                            and var not in exclude)]


    def columns_with_na(self, printing=False):
        """
        Select columns which contain missing values.
        If printing=True, print the percentage of missing values for each column with missing values.
        """
        self.all_cols_with_na = [var for var in self.columns if self.data[var].isnull().sum()>1]
        if printing:
            print("Percentage of all missing values:")
            print()
            print(self.data[self.all_cols_with_na].isnull().mean())


    def cat_columns_with_na(self, printing=False):
        """
        Select categorical columns which contain missing values.
        If printing=True, print the percentage of missing values for each column with missing values.
        """
        self.cat_cols_with_na = [var for var in self.columns if 
                            self.data[var].isnull().sum()>1 and (self.data[var].dtypes == "O")]
        if printing:
            print("Percentage of all missing values:")
            print()
            print(self.data[self.cat_cols_with_na].isnull().mean())

    
    def num_columns_with_na(self, printing=False):
        """
        Select numerical columns which contain missing values.
        If printing=True, print the percentage of missing values for each column with missing values.
        """
        self.num_cols_with_na = [var for var in self.columns if 
                            self.data[var].isnull().sum()>1 and (self.data[var].dtypes != "O")]
        if printing:
            print("Percentage of all missing values:")
            print()
            print(self.data[self.num_cols_with_na].isnull().mean())


    def plot_na_col_target(self, col_na, col_target):
        """
        Group the column col_na with 1/0 and plot the median of col_target for each group.
        """
        df = self.data.copy()
        df[col_na] = np.where(df[col_na].isnull(), 1, 0) # 1 if missing value, 0 otherwise
        df.groupby(col_na)[col_target].median().plot.bar()
        plt.title(col_na)
        plt.show()


    def plot_discrete_col_target(self, col, col_target):
        df = self.data.copy()
        df.groupby(col)[col_target].median().plot.bar()
        plt.title(col)
        plt.ylabel(col_target)
        plt.show()


    def plot_num_col_target(self, col, col_target):
        """
        Scatter plot of the numerical col vs. col_target
        """
        df = self.data.copy()
        plt.scatter(df[col], df[col_target])
        plt.xlabel(col)
        plt.ylabel(col_target)
        plt.show()


    def plot_num_col_hist(self, col):
        """
        Plot histogram of the numerical col
        """
        df = self.data.copy()
        df[col].hist(bins=20)
        plt.ylabel("Frequency")
        plt.title(col)
        plt.show()


    def fill_categorical_na(self, col):
        self.data[col].fillna(value="Missing", inplace=True)


    def fill_numerical_na(self, col, method="mode"):
        if method == "mode":
            mode_val = self.data[col].mode()[0]
            self.data[col].fillna(mode_val, inplace=True)
        if method == "mean":
            mean_val = self.data[col].mean()
            self.data[col].fillna(mean_val, inplace=True)


    def find_frequent_labels(self, col, col_target, rare_perc=0.01):
        temp = self.data.groupby(col)[col_target].count() / len(self.data)
        self.frequent_labels = tmp[temp > rare_perc].index


    def replace_frequent_labels(self, col, col_target, rare_perc=0.01):
        self.data[col] = np.where(self.data[col].isin(self.frequent_labels), self.data[col], "Rare")


    def find_ordinal_categories(self, col, col_target):
        """
        Find discrete values for the strings of the categorical variables,
        so that the smaller value corresponds to the smaller mean of target
        """
        ordered_labels = self.data.groupby(col)[col_target].mean().sort_values().index
        self.ordinal_labels = {k: i for (i,k) in enumerate(ordered_labels,start=0)}


    def replace_ordinal_categories(self, col):
        """
        Assign discrete values to the strings of the categorical variables,
        so that the smaller value corresponds to the smaller mean of target
        """
        self.data[col] = self.data[col].map(self.ordinal_labels)