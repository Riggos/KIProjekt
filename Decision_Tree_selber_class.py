import random
import pandas as pd
import numpy as np


class Decision_Tree_class:

    def __init__(self) -> None:
        pass

    def train_test_split(self, df, test_size):

        if isinstance(test_size, float):
            test_size = round(test_size * len(df))

        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=test_size)

        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)

        return train_df, test_df

    def check_purity(self, data):  # Input is a numpy 2D arry

        label_column = data[:, -1]  # store the values of the label column
        # store all unique values in an arary
        unique_classes = np.unique(label_column)

        if len(unique_classes) == 1:  # check if ther is only on class. Therfore it is unique
            return True
        else:
            return False

    def classify_data(self, data):  # Input is a numpy 2D arry

        label_column = data[:, -1]   # store the values of the label column
        # returns a arry with the classes and one with the count of classes
        unique_classes, counts_unique_classes = np.unique(
            label_column, return_counts=True)

        # returns the class witch appears the most
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]

        return classification

    def get_potential_splits(self, data):  # Input is a numpy 2D arry

        potential_splits = {}
        _, n_columns = data.shape
        # excluding the last column which is the label
        for column_index in range(n_columns - 1):
            values = data[:, column_index]
            unique_values = np.unique(values)  # store the unique values sorted

            potential_splits[column_index] = unique_values

        return potential_splits

    # check if the given data is below or above a given split_value
    def split_data(self, data, split_column, split_value):

        # store all values of a chosen column
        split_column_values = data[:, split_column]

        # only choose the values that are above the split_value
        data_below = data[split_column_values <= split_value]
        # only choose the values that are below the split_value
        data_above = data[split_column_values > split_value]

        return data_below, data_above

    def calculate_entropy(self, data):

        label_column = data[:, -1]
        # count the unique values for each class
        _, counts = np.unique(label_column, return_counts=True)

        # calculate the probability for each class
        probabilities = counts / counts.sum()
        # calculate the entropy
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy

    def calculate_overall_entropy(self, data_below, data_above):

        # count the total number of values
        n = len(data_below) + len(data_above)
        # count the total number of the data_below
        p_data_below = len(data_below) / n
        # count the total number of the data_above
        p_data_above = len(data_above) / n

        overall_entropy = (p_data_below * self.calculate_entropy(data_below)  # calculate the total entropy
                           + p_data_above * self.calculate_entropy(data_above))

        return overall_entropy

    # calculate all the entropies for the potential splits and store the lowest one and the responding column and the split value

    def determine_best_split(self, data, potential_splits):

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

    def decision_tree_algorithm(self, df, counter=0, min_samples=2, max_depth=5):

        # data preparations
        if counter == 0:
            global COLUMN_HEADERS, FEATURE_TYPES
            COLUMN_HEADERS = df.columns
            FEATURE_TYPES = self.determine_type_of_feature(df)
            data = df.values
        else:
            data = df

        # base cases
        if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
            classification = self.classify_data(data)

            return classification

        # recursive part
        else:
            counter += 1

            # helper functions
            potential_splits = self.get_potential_splits(data)
            split_column, split_value = self.determine_best_split(
                data, potential_splits)
            data_below, data_above = self.split_data(
                data, split_column, split_value)

            # check for empty data
            if len(data_below) == 0 or len(data_above) == 0:
                classification = self.classify_data(data)
                return classification

            # determine question
            feature_name = COLUMN_HEADERS[split_column]
            type_of_feature = FEATURE_TYPES[split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            # feature is categorical
            else:
                question = "{} = {}".format(feature_name, split_value)

            # instantiate sub-tree
            sub_tree = {question: []}

            # find answers (recursion)
            yes_answer = self.decision_tree_algorithm(
                data_below, counter, min_samples, max_depth)
            no_answer = self.decision_tree_algorithm(
                data_above, counter, min_samples, max_depth)

            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base case).
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree

    def classify_example(self, example, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")

        # ask question
        if comparison_operator == "<=":  # feature is continuous
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        # feature is categorical
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        # base case
        if not isinstance(answer, dict):
            return answer

        # recursive part
        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)

    def calculate_accuracy(self, df, tree):

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

        # ask question
        if comparison_operator == "<=":
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        # feature is categorical
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        # base case
        if not isinstance(answer, dict):
            return answer

        # recursive part
        else:
            residual_tree = answer
            return self.predict_example(example, residual_tree)
