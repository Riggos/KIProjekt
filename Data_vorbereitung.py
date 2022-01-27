import pandas as pd
import seaborn as sns
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Data_preparation:

    def __init__(self) -> None:

        pass

    def load_csv_to_df(self, filepath):

        df = pd.read_csv(filepath)  # load Dataset

        if(df.columns[0] == 'Unnamed: 0'):

            df = df.iloc[:, 1:]  # only use the feature and Label Colloums

        return df

    def load_data_to_list(self, filepath):
        df = pd.read_csv(filepath)  # load Dataset

        if(df.columns[0] == 'Unnamed: 0'):

            df = df.iloc[:, 1:]  # only use the feature and Label Colloums

        dataset = df.values.tolist()
        return dataset

    def split_data(self, df, test_size):

        just_a_list = []
        listindex = False

        if isinstance(test_size, float):
            test_size = round(test_size * len(df))

        if type(df) == type(just_a_list):

            listindex = True
            df = pd.DataFrame(df)

        indices = df.index.tolist()
        # get random indizes for the split
        test_indices = random.sample(population=indices, k=test_size)
        X = df.iloc[:, :-1]  # get the featuers
        Y = df.iloc[:, -1]  # only get the Label column

        X_train = X.loc[test_indices]  # get the training feature
        y_train = Y.loc[test_indices]  # get the training Labels
        X_test = X.drop(test_indices)  # get the test features
        y_test = Y.drop(test_indices)  # get the training Labels

        if listindex:
            X_train = X_train.values.tolist()
            X_test = X_test.values.tolist()
            y_train = y_train.values.tolist()
            y_test = y_test.values.tolist()

        return X_train, X_test, y_train, y_test

    def get_potential_splits(self, data):  # Input is a numpy 2D arry

        potential_splits = {}
        _, n_columns = data.shape
        # excluding the last column which is the label
        for column_index in range(n_columns - 1):
            values = data[:, column_index]
            unique_values = np.unique(values)  # store the unique values sorted

            potential_splits[column_index] = unique_values

        return potential_splits

    def draw_scatterplot(self, dataset, train_df, x_value, y_value, feature_to_split):
        potential_splits = self.get_potential_splits(dataset.values)
        sns.lmplot(data=train_df, x=x_value, y=y_value, fit_reg=False, hue="Label",
                   size=6, aspect=1.5)  # Plot a scatterplot for two chosen columns
        # draw the potential Splits for the chosen row
        plt.vlines(x=potential_splits[feature_to_split], ymin=7, ymax=80)
