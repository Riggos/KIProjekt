"""
Dieses Python File beinhaltet eine Klasse "Random_Forrest_class" welche einen Merkmaldatensatz und den dazugehörigen Labels eine Klassifikation nach einem 
Random Forrest Verfahren durchführt

"""

__author__ = "Eric Hirsch und Jonas Morsch"
__version__ = "1.0.0"


import random
import pandas as pd
import numpy as np


class Random_Forrest_class:

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

    def get_potential_splits(self, data, random_subspace):
        """Bekommt einen 2D-numpy Array
        Schaut sich die gesamten Daten an und gibt zurück wo die Daten getrennt werden
        können.
        "random_subspace": Wieviele Eigenschaften werde zufällig genommen
         --> potenzielle Trennpunkte"""
        potential_splits = {}
        _, n_columns = data.shape

        column_indicies = list(range(n_columns-1))

        if random_subspace:

            column_indicies = random.sample(
                population=column_indicies, k=random_subspace)

        for column_index in column_indicies:
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

        data_below = data[split_column_values <= split_value]

        data_above = data[split_column_values > split_value]

        return data_below, data_above

    def calculate_entropy(self, data):
        """Rechnet für die übergebenen Daten "data" die Entropie aus"""

        label_column = data[:, -1]

        _, counts = np.unique(label_column, return_counts=True)

        probabilities = counts / counts.sum()

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

    # Berechner der Entrophie um mögliche unterteilung der Daten zu finden speicher Spalte und den Wert zum Teilen des Datensatzes

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

    def decision_tree_algorithm(self, df, counter=0, min_samples=2, max_depth=5, random_subspace=None):
        """Erstellt den Entscheidungsbaum:
        "counter" gibt an wie oft der Algoitmus schon durchlaufen wurde
        "min_samples": alles was gleich oder kleiner ist wird direkt klassifiziert
        "max-depth": gibt an wie Tief der Baum sein soll
        "random_subspace": WIeviele Eigenschaften werden zufällig genommen
        """
        if counter == 0:
            global COLUMN_HEADERS, FEATURE_TYPES
            COLUMN_HEADERS = df.columns
            FEATURE_TYPES = self.determine_type_of_feature(df)
            data = df.values
        else:
            data = df

        if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
            classification = self.classify_data(data)

            return classification

        else:
            counter += 1

            potential_splits = self.get_potential_splits(data, random_subspace)
            split_column, split_value = self.determine_best_split(
                data, potential_splits)
            data_below, data_above = self.split_data(
                data, split_column, split_value)

            if len(data_below) == 0 or len(data_above) == 0:
                classification = self.classify_data(data)
                return classification

            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            else:
                question = "{} = {}".format(feature_name, split_value)

            sub_tree = {question: []}

            yes_answer = self.decision_tree_algorithm(
                data_below, counter, min_samples, max_depth, random_subspace)
            no_answer = self.decision_tree_algorithm(
                data_above, counter, min_samples, max_depth, random_subspace)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree

    def classify_example(self, example, tree):
        """Bekommt ein Datum und versucht daraus die Klasse zu bestimmen"""
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
            return self.classify_example(example, residual_tree)

    def calculate_accuracy(self, df, tree):
        """Berechnet die Genauigkeit des Entscheidungsbaum anhand eines Datensets"""
        df["classification"] = df.apply(
            self.classify_example, axis=1, args=(tree,))
        df["classification_correct"] = df["classification"] == df["Label"]

        accuracy = df["classification_correct"].mean()

        return accuracy

    def decision_tree_predictions(self, test_df, tree):
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

    def calculate_accuracy2(self, predictions, labels):
        predictions_correct = predictions == labels
        accuracy = predictions_correct.mean()

        return accuracy

    def bootstrapping(self, train_df, n_bootstrap):
        """Entnimmt aus dem Hauptdaten (traind_df) zufällige einzelne Daten und erstellt daraus "neue"
        Trainingsdaten (Bootstrapped) --> jeweils für einen random forest
        n_bootstrap: Gibt die Größe an"""
        bootstraps_indices = np.random.randint(
            low=0, high=len(train_df), size=n_bootstrap)
        df_bootstrapped = train_df.iloc[bootstraps_indices]

        return df_bootstrapped

    def random_forrest_algorithm(self, train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
        """Erstellt den Random Forest
        "n_trees": Wieviele Entscheidungsbäume sollen entstehen
        "n_bootstrap": Größe des Bootstrap Trainingsdaten
        "n_features": Mit wievielen Features soll ein Entscheidungsbaum entscheiden
        "dt_max_depth": maximale Tiefe der Entscheidungsbäume
        """
        forest = []
        # Erstellt die Entscheidungsbäume
        for i in range(n_trees):
            df_bootstrapped = self.bootstrapping(train_df, n_bootstrap)
            tree = self.decision_tree_algorithm(
                df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
            forest.append(tree)
        # Gibt alle erstellten Entscheidungsbäume zurück
        return forest

    def random_forrest_predictions(self, test_df, forest):
        """Führt die Bestimmung der Klasse mit random forest durch"""
        df_predictions = {}
        # Sammelt alle Vorhersagen in ein Dictinary
        for i in range(len(forest)):
            column_name = "tree{}".format(i)
            predictions = self.decision_tree_predictions(
                test_df, tree=forest[i])
            df_predictions[column_name] = predictions

        df_predictions = pd.DataFrame(df_predictions)
        # Zählt die Vorhersagen der verschiedenen Bäume zusammen --> die am meisten
        # vorhergesagte Klasse wird genommen
        random_forest_predicitons = df_predictions.mode(axis=1)[0]

        return random_forest_predicitons
