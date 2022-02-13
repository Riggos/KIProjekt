"""
Dieses Python File beinhaltet eine Klasse "Decison_Tree_Class" welche einen Merkmaldatensatz und den dazugehörigen Labels eine Klassifikation nach einem 
Decision Tree Verfahren durchführt.

Ziel ist es durch Bedingungen die Daten solagen aufzuteilen bis ein Datenanteil nur noch
eine Klasse enthält. Dadurch kann dann die Klasse von Datensätzen bestimmt werden

"""

__author__ = "Eric Hirsch und Jonas Morsch"
__version__ = "1.0.0"


import random
import pandas as pd
import numpy as np


class Decision_Tree_class:

    def __init__(self) -> None:
        pass

    def train_test_split(self, df, test_size):
        """Trennt die Daten zu Traings und Test Daten.
        Diese weden als panda_frame abgespeichert"""

        if isinstance(test_size, float):
            test_size = round(test_size * len(df))

        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=test_size)

        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)

        return train_df, test_df

    def check_purity(self, data):
        """Schaut ob die geteilten Daten durch den Entscheidungsbaum nur eine 
        Klasse enthält"""

        label_column = data[:, -1]

        unique_classes = np.unique(label_column)

        if len(unique_classes) == 1:
            return True
        else:
            return False

    def classify_data(self, data):
        """Gibt an zu welcher Klasse die geteilten Daten gehören. Auch wenn die Daten 
        nicht "pure" sind --> es wird größte Anzahl einer Klasse genommen im geteilten
        Datensatz genommen"""

        label_column = data[:, -1]

        unique_classes, counts_unique_classes = np.unique(
            label_column, return_counts=True)

        index = counts_unique_classes.argmax()
        classification = unique_classes[index]

        return classification

    def get_potential_splits(self, data):
        """Bekommt einen 2D-numpy Array
        Schaut sich die gesamten Daten an und gibt zurück wo die Daten getrennt werden
        können.
         --> potenzielle Trennpunkte"""

        potential_splits = {}
        _, n_columns = data.shape

        for column_index in range(n_columns - 1):
            values = data[:, column_index]
            unique_values = np.unique(values)

            potential_splits[column_index] = unique_values

        return potential_splits

    def split_data(self, data, split_column, split_value):
        """Trennt die Daten an einem bestimmten Ort auf:
            split_column: Welches Merkmal?
            split_value: Bedingung wo die Daten im definerte Merkmal aufgeteilt werden
            Gibt die beiden aufgeteilten Daten zurück"""

        split_column_values = data[:, split_column]
        type_of_feature = FEATURE_TYPES[split_column]

        if type_of_feature == "continuous":
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values > split_value]
        else:
            data_below = data[split_column_values == split_value]
            data_above = data[split_column_values != split_value]

        return data_below, data_above

    def calculate_entropy(self, data):
        """Rechnet für die übergebenen Daten "data" die Entropie aus"""

        # die Label Spalte auswählen
        label_column = data[:, -1]

        # Zählt die Häfuigkeit der einzelnen Klassen
        _, counts = np.unique(label_column, return_counts=True)

        # Rechnet die Wahrscheinlichkeiten der Klassen aus
        probabilities = counts / counts.sum()
        # Rechnet die Entropie aus
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy

    def calculate_overall_entropy(self, data_below, data_above):
        """Bekommt die in zwei Teilen aufgeteilen Daten und rechnet für beide die Entropie
        aus und berchnet daraus die Overall Entropy
        --> wie gut ist die Aufteilung"""

        n = len(data_below) + len(data_above)

        p_data_below = len(data_below) / n

        p_data_above = len(data_above) / n

        overall_entropy = (p_data_below * self.calculate_entropy(data_below)
                           + p_data_above * self.calculate_entropy(data_above))

        return overall_entropy

    def determine_best_split(self, data, potential_splits):
        """Findet die eine beste Aufteilung für die Daten bzw. Eigenschaften"""

        overall_entropy = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(
                    data, split_column=column_index, split_value=value)
                current_overall_entropy = self.calculate_overall_entropy(
                    data_below, data_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value

    def determine_type_of_feature(self, df):
        """Schaut ob die Features kategorisch (Bsp.: Geschlecht) oder koninuierlich (Bsp.: Größe) sind"""

        feature_types = []
        # Bei Zahlen: bei vielen einzigartigen Zahlen ist es kontinuierlich
        # Bei Strings: immer kategorisch
        n_unique_values_treshold = 15
        for feature in df.columns:
            if feature != "label":
                unique_values = df[feature].unique()
                example_value = unique_values[0]

                if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                    feature_types.append("categorical")
                else:
                    feature_types.append("continuous")

        return feature_types

    def decision_tree_algorithm(self, df, counter=0, min_samples=2, max_depth=5):
        """Erstellt den Entscheidungsbaum:
        "counter" gibt an wie oft der Algoitmus schon durchlaufen wurde
        "min_samples": alles was gleich oder kleiner ist wird direkt klassifiziert
        "max-depth": gibt an wie Tief der Baum sein soll
        """

        # Wenn der Alorithmus zum erstmal Aufgerufen wird --> extrahiere die wichtigen Daten aus dem Dataframe
        if counter == 0:
            global COLUMN_HEADERS, FEATURE_TYPES
            COLUMN_HEADERS = df.columns
            FEATURE_TYPES = self.determine_type_of_feature(df)
            data = df.values
        else:
            data = df

        # Bricht den Algorithmus ab wenn die Daten "pure" sind oder die Tiefe des Baums erreicht ist oder die Anzahl an Daten zu klein ist
        if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
            classification = self.classify_data(data)

            return classification

        # Ruft die Funktion rekursiv auf um den Baum zu erstellen
        else:
            counter += 1

            # Ruft oben definerte Funktionen auf --> Teilt die Daten auf: Vorbereitung für die Äste und Blätter
            potential_splits = self.get_potential_splits(data)
            split_column, split_value = self.determine_best_split(
                data, potential_splits)
            data_below, data_above = self.split_data(
                data, split_column, split_value)

            if len(data_below) == 0 or len(data_above) == 0:
                classification = self.classify_data(data)
                return classification

            # inter Werte mit den richitnge Feature names ersetzen
            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]

            # Baut ein Dictonary auf indem der Entscheidungsbaum enthalten ist
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            else:
                question = "{} = {}".format(feature_name, split_value)

            sub_tree = {question: []}

            # Ruft für jede neue Verzweigung (Bedingung: Ja und Nein) den Algorithmus mit den
            # entsprechenen aufgeteilten Daten neu auf (rekursiv)
            yes_answer = self.decision_tree_algorithm(
                data_below, counter, min_samples, max_depth)
            no_answer = self.decision_tree_algorithm(
                data_above, counter, min_samples, max_depth)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree

    def classify_example(self, example, tree):
        """Bekommt ein Datum und versucht daraus die Klasse zu bestimmen"""
        question = list(tree.keys())[
            0]  # liefert das erste Elemnt aus einer Liste aus Fragen
        feature_name, comparison_operator, value = question.split(" ")

        if comparison_operator == "<=":  # feature is continuous
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
    # Wenn die Antwort kein Dictionary ist haben wir unsere Klassifizierung
        if not isinstance(answer, dict):
            return answer

        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)

    def calculate_accuracy(self, df, tree):
        """Berechnet die Genauigkeit des Entscheidungsbaum anhand eines Datensets"""

        df["classification"] = df.apply(
            self.classify_example, axis=1, args=(tree,))
        df["classification_correct"] = df["classification"] == df["Label"]

        accuracy = df["classification_correct"].mean()

        return accuracy

    def decision_tree_predictions(self, test_df, tree):
        """"""

        predictions = test_df.apply(self.predict_example, args=(tree,), axis=1)
        return predictions

    def predict_example(self, example, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")

        if comparison_operator == "<=":
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer

        else:
            residual_tree = answer
            return self.predict_example(example, residual_tree)
