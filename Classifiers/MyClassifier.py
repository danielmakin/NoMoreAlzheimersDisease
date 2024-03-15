from abc import abstractmethod
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
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

    def display_hyperparameter_results(self, grid_search, parameters, param_grid, file_name):
        results = pd.DataFrame(grid_search.cv_results_)
        rows = math.ceil(len(parameters) / 2)

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(20, rows * 4))
        fig.suptitle("Trends for the change of Hyper-Parameters")

        for i, ax in enumerate(axes.flat):
            if i == len(parameters):
                break
            print("Here is the PARAM: " + parameters[i])
            temp = results.groupby('param_' + str(parameters[i]))['mean_test_score']
            ax.plot(param_grid[parameters[i]], temp.mean(), color='blue', marker='o')
            ax.errorbar(param_grid[parameters[i]], temp.mean(), yerr=temp.std(), color='orange')
            ax.set_xlabel(parameters[i] + " value")
            ax.set_ylabel("Accuracy")

            if parameters[i] == 'C':
                ax.set_xscale('log')

        plt.tight_layout()
        # plt.show()

        if file_name != "":
            # Save the File
            plt.savefig(file_name + "_scores.png")

        results = pd.DataFrame(grid_search.cv_results_)
        rows = math.ceil(len(parameters) / 2)

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(20, rows * 4))
        fig.suptitle("Trends for the change of Hyper-Parameters in Terms of Time")

        for i, ax in enumerate(axes.flat):
            if i == len(parameters):
                break
            temp = results.groupby('param_' + parameters[i])['mean_fit_time']
            ax.plot(param_grid[parameters[i]], temp.mean(), color='blue', marker='o')
            ax.errorbar(param_grid[parameters[i]], temp.mean(), yerr=temp.std(), color='orange')
            ax.set_xlabel(parameters[i] + " value")
            ax.set_ylabel("Time Taken (s)")

        plt.tight_layout()
        # plt.show()

        if file_name != "":
            # Save the File
            plt.savefig(file_name + "_times.png")

    def display_results_against_iterations(self, results):
        # Extract the mean test scores and number of iterations
        mean_test_scores = results['mean_test_score']
        num_iterations = np.arange(1, len(mean_test_scores) + 1)

        # Track maximum accuracy found at any iteration
        max_accuracy = np.maximum.accumulate(mean_test_scores)

        # Plot maximum accuracy against number of iterations
        plt.plot(num_iterations, max_accuracy, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Maximum Test Accuracy')
        plt.title('Maximum Accuracy vs. Number of Iterations')
        plt.grid(True)

    def display_class_results_text(self, y_test, y_true, auc):
        print("Accuracy is " + str(accuracy_score(y_true, y_test)))
        # Get the recall and metrics for each class
        recall = list(recall_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))
        precision = list(precision_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))
        f1 = list(f1_score(y_true, y_test, average=None, labels=['SCD', 'MCI', 'AD']))

        data = [["Recall"] + recall]
        data.append(["Precision"] + precision)
        data.append(["F1 Score"] + f1)
        data.append(["AUC"] + auc)



        table = tabulate(data, ['', 'SCD', 'MCI', 'AD'], tablefmt="grid")
        print(table)

        cm = confusion_matrix(y_true, y_test, labels=['SCD', 'MCI', 'AD'])

        # Display confusion matrix using seaborn heatmap
        print(cm)

    def auc_scores_svm(self, X_train, y_train, X_test, y_test):

        # Calibrate the classifier using Platt scaling
        calibrated_svc = CalibratedClassifierCV(SVC(probability=True))
        calibrated_svc.fit(X_train, y_train)

        # Get calibrated probabilities
        y_pred_proba = calibrated_svc.predict_proba(X_test)

        # Calculate AUC for each class using One-vs-Rest (ovr)
        auc_ovr = roc_auc_score(y_test, y_pred_proba, average=None, multi_class='ovr')
        print(calibrated_svc.classes_)
        return list(np.round(auc_ovr, 3))[::-1]
