from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import metrics


def loadCsv(filename):

    df = pd.read_csv(filename, usecols=['Rel_BreitGroß', 'RelSpitze_oben',
                     'RelSpitze_unten', 'Anzahl_Linie', 'Anzahl_Kreis', 'Label'])
    dataset = df.values.tolist()

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


lines = loadCsv('ric.csv')
dataset = pd.DataFrame(lines)
# print(dataset.head())
data = dataset.iloc[:, 0:5]
# print(data.head())
label = dataset.iloc[:, 5]
# print(data.head())


X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.8, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
