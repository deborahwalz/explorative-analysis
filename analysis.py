import numpy as np
import pandas as pd

class Analysis():
    """
    Class for first explorative data analysis.
    """
    def __init__(self, data):
        self.data = data
        self.columns = data.columns

    def columns_with_na(self, printing=False):
        self.cols = [var for var in self.columns if self.data[var].isnull().sum() > 1]

        if printing:
            print("Percentage of missing values:")
            print()
            print(self.data[self.cols].isnull().mean())
        
