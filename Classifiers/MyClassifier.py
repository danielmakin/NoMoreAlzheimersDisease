from abc import abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from tabulate import tabulate

class MyClassifier:
    def getXy(self, df):
        '''
        Seperate the label from the feature data.
        
        Args : 
            df (DataFrame) : The data to be seperated
        Returns :
            X (DataFrame) : The feature data.
            y (list) : The labels associated.'''
        
        # Independant
        X = df.drop("DX", axis=1).values
        # Dependant
        y = df["DX"].values

        return X, y
    @abstractmethod
    def hyper_parameter_selection(self, verbose=0):
        pass
    @abstractmethod
    def test(self, metrics):
        pass

    def display_hyperparameter_results(self, grid_search, parameters, param_grid):
        results = pd.DataFrame(grid_search.cv_results_)

        fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
        fig.suptitle("Trends for the change of Hyper-Parameters")

        for i in range(len(parameters)):
            temp = results.groupby('param_' + parameters[i])['mean_test_score']
            plt.subplot(2, 2, i+1)
            plt.plot(param_grid[parameters[i]], temp.mean(), color='blue', marker='o')
            plt.errorbar(param_grid[parameters[i]], temp.mean(), yerr=temp.std(), color='orange')
            plt.xlabel(parameters[i] + " value")
            plt.ylabel("Accuracy")

            if parameters[i] == 'C':
                plt.xscale('log')

        plt.show()

    def display_class_results_text(self, y_test, y_true):
        print("Accuracy is " + str(accuracy_score(y_true, y_test)))
        # Get the recall and metrics for each class
        recall = list(recall_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))
        precision = list(precision_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))
        f1 = list(f1_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))

        data = [["Recall"] + recall]
        data.append(["Precision"] + precision)
        data.append(["F1 Score"] + f1)



        table = tabulate(data, ['', 'SCD', 'MCI', 'AD'], tablefmt="grid")
        print(table)


# TODO :: Comment and fix the fact that there is no x label for top grpahs
# TODO :: See if the loss function cab be printed
