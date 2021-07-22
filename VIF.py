from dataload import Dataset

import numpy as np
import pandas as pd


from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from imblearn.pipeline import Pipeline


class VIF_filter():
    def __init__(self,train_data,origin_columns,threshold=4):
        self.threshold=threshold
        self.train_data=train_data
        self.origin_columns =origin_columns
        self.remain_columns=list(self.train_data.columns)
        self.filtered_columns=[]
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
        ])
        self.train_data = pipe.fit_transform(self.train_data)
        self.train_data = add_constant(self.train_data)


    def filtering(self):
        print("Start VIF filtering--------------------------------")
        while (1):
            vif_df= np.array([variance_inflation_factor(self.train_data, i) for i in range(self.train_data.shape[1])])
            VIF_argmax_index = vif_df[1:].argmax()
            if vif_df[1:].max()< self.threshold:
                break
            self.filtered_columns.append(self.origin_columns[VIF_argmax_index])
            self.remain_columns.pop(VIF_argmax_index)
            self.train_data = np.delete(self.train_data,VIF_argmax_index+1,1)
            print(f'{self.origin_columns[VIF_argmax_index]} is filtered out')
        print("---------------------------------------------------")
        print()
        return self.remain_columns , self.filtered_columns

if __name__ == "__main__":
    csv_directory_fname = './data/sample.csv'
    data = Dataset(csv_directory_fname, 'Cardio')
    xtrain, xtest, ytrain, ytest, columns = data.load_dataset()


    VIF=VIF_filter(xtrain,columns,4)
    remain_columns, filtered_columns = VIF.filtering()