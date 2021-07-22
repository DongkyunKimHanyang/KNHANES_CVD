import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



class Dataset(): #Data load Class
    def __init__(self,csv_directory_fname,label):
        self.df_csv=pd.read_csv(csv_directory_fname)
        self.label=label

    def drop_columns(self, drop_list):
        self.df_csv = self.df_csv.drop(columns=drop_list)

    def load_dataset(self, drop_VIF=False,drop_Boruta=False):  # Data load function
        self.df_csv.dropna(subset=[self.label], inplace=True)

        y = self.df_csv.loc[:, self.label]


        columns = np.array(list(self.df_csv)[:-1])
        train, test = train_test_split(self.df_csv, test_size=0.2, stratify=self.df_csv.loc[:, self.label])
        train.reset_index(inplace=True,drop=True)
        test.reset_index(inplace=True,drop=True)
        xtrain = train.iloc[:, :-1]
        xtest = test.iloc[:, :-1]
        ytrain = train.loc[:, self.label]
        ytest = test.loc[:, self.label]
        return xtrain, xtest, ytrain, ytest, columns


if __name__=="__main__":
    csv_directory='./data/sample.csv'
    data=Dataset(csv_directory,'Cardio')
    xtrain, xtest, ytrain, ytest, columns = data.load_dataset(effect_coding=False, log_transform=False,drop_VIF=True,drop_Boruta=True)
