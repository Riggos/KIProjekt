import pandas as pd
import seaborn as sns


def show_Data_information(file_path):
    df = pd.read_csv(file_path)  # load Dataset

    if(df.columns[0] == 'Unnamed: 0'):

        df = df.iloc[:, 1:]  # only use the feature and Label Colloums

    print(df.dtypes)
    print("\n\n")
    print(df.head(10))
    print("\n\n")

    for member in df.columns:
        print("missing values", member, ":", len(df.loc[(df[member] == '?')]))

    corr = df.iloc[:, :-1].corr()
    sns.heatmap(corr)


def Split_Data(df, splitratio, list=False):

    trainsize = int(len(df)*splitratio)

    if list == False:
        print("DataType:Pandas Datafram")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X = df.values.tolist()
        Y = df.values.tolist()
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for index in range(0, trainsize):
            X_train.append(X[index])
            y_train.append(y[index])
        for index in range(0, (len(X)-trainsize)):
            X_test.append(X[index])
            y_test.append[y[index]]

        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test
