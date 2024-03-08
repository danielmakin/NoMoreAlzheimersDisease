import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
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

    def test(self, metrics=False, max_iterations=1000, verbose=0):

        y = self.df.pop("DX")
        X = self.df

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the MLP classifier
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iterations, random_state=42, verbose=verbose)
        mlp_classifier.fit(X_train_scaled, y_train)

        # Predict the labels for the test set
        y_pred = mlp_classifier.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        self.mlp_classifier = mlp_classifier

        if metrics == True:
            print(classification_report(y_test, y_pred))

    def plot_loss(self):
        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.mlp_classifier.loss_curve_) + 1), self.mlp_classifier.loss_curve_, linestyle='-')
        plt.title('Training Loss Curve of MLP Classifier')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()