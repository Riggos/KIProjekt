"""Dieses Python File beinhaltet eine Klasse um Data Augmentation auf Bilder in einem Ordner durchzuführen
und diese mit den ursprünglichen Bildern in einen angegebenden Ort abzuspeichern
"""

__author__ = ""
__version__ = "1.0.1"


<<<<<<< HEAD
=======

>>>>>>> acbedfe7fc982edf5d30f9d7f109226b4a6d5cc3
from tkinter import Image
import pandas as pd
import seaborn as sns
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import xlsxwriter
from openpyxl import Workbook, load_workbook
import os.path
from openpyxl.utils.dataframe import dataframe_to_rows


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

    def split_data(self, df, train_size):

        just_a_list = []
        listindex = False

        if isinstance(train_size, float):
            train_size = round(train_size * len(df))

        if type(df) == type(just_a_list):

            listindex = True
            df = pd.DataFrame(df)

        indices = df.index.tolist()
        # get random indizes for the split
        train_indices = random.sample(population=indices, k=train_size)
        X = df.iloc[:, :-1]  # get the featuers
        Y = df.iloc[:, -1]  # only get the Label column

        X_train = X.loc[train_indices]  # get the training feature
        y_train = Y.loc[train_indices]  # get the training Labels
        X_test = X.drop(train_indices)  # get the test features
        y_test = Y.drop(train_indices)  # get the training Labels

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

    def draw_scatterplot(self, df):
        di = {0: 'Dosenoeffner', 1: 'Falschenoeffner',
              2: 'Korkenzieher'}  # dictionary
        columns = df.columns
        before = sns.pairplot(df.replace({columns[-1]: di}), hue=columns[-1])
        before.fig.suptitle('Pair Plot of Dataset', y=1.08)

    def create_Image_dataset(self, img_folder, IMG_HEIGHT, IMG_WIDTH):

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

    def draw_corr_matrix(self, df):
        features = df.iloc[:, :-1]
        CorrMatrix = features.corr()
        sns.heatmap(CorrMatrix, annot=True)

    def classification_report_csv(self, report, path):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv("Classification_reports/"+path, index=False)
