import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import numpy as np
from Classifiers.MyClassifier import MyClassifier


class MyMLP(MyClassifier):
    def __init__(self, file_path, fields):
        # Get the .csv and remove the unnecessary fields
        self.df = pd.read_csv(file_path)[fields]
        # Now we can remove the null values
        self.df.dropna(inplace=True)

    def test(self, metrics=False, max_iterations=10000, verbose=0):

        # y = self.df.pop("DX")
        # X = self.df

        # X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # # Standardize features by removing the mean and scaling to unit variance
        # scaler = StandardScaler()
        # self.X_train_scaled = scaler.fit_transform(X_train)
        # self.X_test_scaled = scaler.transform(X_test)

        # Initialize and train the MLP classifier
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes = self.grid_search.best_params_['hidden_layer_sizes'],
            activation = self.grid_search.best_params_['activation'],
            solver = self.grid_search.best_params_['solver'],
            alpha = self.grid_search.best_params_['alpha'],
            max_iter = max_iterations
        )
        mlp_classifier.fit(self.X_train_scaled, self.y_train)

        # Predict the labels for the test set
        y_pred = mlp_classifier.predict(self.X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)

        self.mlp_classifier = mlp_classifier

        if metrics == True:
            print(classification_report(self.y_test, y_pred))

            print(confusion_matrix(self.y_test, y_pred, labels=['SCD', 'MCI', 'AD']))

    def plot_loss(self, file_name):
        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.mlp_classifier.loss_curve_) + 1), self.mlp_classifier.loss_curve_, linestyle='-')
        plt.title('Training Loss Curve of MLP Classifier')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(file_name)

    def hyper_parameter_selection(self, verbose=0, file_name=""):
        y = self.df.pop("DX")
        X = self.df

        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        mlp = MLPClassifier(max_iter=10000, verbose=verbose)

        self.parameter_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 100), (5, 10)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05]
        }

        self.parameters = ['hidden_layer_sizes', 'activation', 'solver', 'alpha']

        clf = RandomizedSearchCV(mlp, self.parameter_grid, n_jobs=-1, cv=3, n_iter=72)

        clf.fit(X_train, self.y_train)

        print("Best Parameters: ", clf.best_params_)
        print("Best Training Score: ", clf.best_score_)

        self.grid_search = clf
        #super().display_hyperparameter_results(grid_search=self.grid_search, parameters=self.parameters, param_grid=self.parameter_grid, file_name=file_name)

