from tkinter import Image
import pandas as pd
import seaborn as sns
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2


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

        y_train = y_train.to_frame()
        y_test = y_test.to_frame()

        print(
            f'training set size: {X_train.shape[0]} samples \ntest set size: {X_test.shape[0]} samples')

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

    def load_pictures(self, folder_location):
        path = 'animal/'
        img_dict = dict()

        for root, dirs, files in os.walk(path):
            print(os.path.basename(root))
            my_key = os.path.basename(root)

            dir_images = []
            for file_ in files:
                full_file_path = os.path.join(root, file_)
                img = cv2.imread(full_file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dir_images.append(img)

            img_dict[my_key] = dir_images

        return img_dict

    def create_dataset(self, img_folder, IMG_HEIGHT, IMG_WIDTH):

        img_data_array = []
        class_name = []

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):

                image_path = os.path.join(img_folder, dir1,  file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(
                    image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name

    def split_Image_data(self, Images, Labels, train_size):

        index_list = []  # an empty list for the indicies
        test_index = []
        X_test = []
        X_train = []
        y_test = []
        y_train = []

        if isinstance(train_size, float):
            test_size = round(train_size * len(Images))

        for i in range(len(Images)):
            index_list.append(i)

        # get random indizes for the split
        train_indices = random.sample(population=index_list, k=test_size)
        for i in range(len(Images)):
            if i not in train_indices:
                test_index.append(i)

        for index in train_indices:
            X_train.append(Images[index])
            y_train.append(Labels[index])

        for index in test_index:
            X_test.append(Images[index])
            y_test.append(Labels[index])

        X_train = np.asarray(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, X_test, y_train, y_test
