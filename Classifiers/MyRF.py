from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from Classifiers.MyClassifier import MyClassifier
from imblearn.over_sampling import SMOTE


class MyRF(MyClassifier):
    def __init__(self, file_name, fields_to_drop):
        self.df = pd.read_csv(file_name).drop(fields_to_drop, axis=1)

    def hyper_parameter_selection(self, iterations = 100, verbose=0):
        # Create the test object
        self.rf = RandomForestClassifier()

        X, y = self.getXy(self.df)

        # Split for a 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Now we create the grid of params we need to test
        self.param_grid = {
            'n_estimators' : [50, 100, 150, 200, 250, 300],
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'bootstrap' : [True, False],
            'class_weight' : ['balanced', 'balanced_subsample'],
            'max_depth' : [10, 20, 30],
            'min_samples_split' : [2, 5, 10],
            'max_features' : ['sqrt', 'log2']
        }

        # Create the test Object and fit
        grid_search = RandomizedSearchCV(estimator=self.rf, param_distributions=self.param_grid, cv=3, scoring='accuracy', n_iter=iterations, verbose=verbose)
        grid_search.fit(self.X_train, self.y_train)

        self.parameters = ['n_estimators', 'criterion', 'bootstrap', 'class_weight', 'max_depth', 'min_samples_split', 'max_features']
        # Display the best parameters and best accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        # Store these values for next function

        self.grid_search = grid_search

        super().display_hyperparameter_results(self.grid_search, self.parameters, self.param_grid)
        super().display_results_against_iterations(self.grid_search.cv_results_)

    def train_test_SMOTE(self, metrics=True):
        test_classifier = RandomForestClassifier(
            bootstrap = self.grid_search.best_params_['bootstrap'],
            n_estimators = self.grid_search.best_params_['n_estimators'],
            max_depth = self.grid_search.best_params_['max_depth'],
            class_weight = self.grid_search.best_params_['class_weight'],
            min_samples_split = self.grid_search.best_params_['min_samples_split'],
            max_features = self.grid_search.best_params_['max_features'],
            criterion = self.grid_search.best_params_['criterion'],
            random_state = 42 # Ensure reproducibility
        )

        # Now We SMOTE The Training Data
        sm = SMOTE(random_state=42)

        # Oversample the minority classes
        X_train, y_train = sm.fit_resample(self.X_train, self.y_train)

        # Runs a test to find the accuracy of the model
        test_classifier.fit(X_train, y_train)

        y_testresult = test_classifier.predict(self.X_test)

        # Now display the metrics if needed
        if metrics == True:
            super().display_class_results_text(y_testresult, self.y_test, self.get_auc(X_train, y_train, self.X_test, self.y_test))

        # Pre-Split the Data
        # Check that past me has done SMOTE correctly.
        # Make Differences between training classes be seen

        
    
    def get_auc(self, X_train, y_train, X_test, y_test):
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)

        # Predict probabilities on the validation set
        y_pred_proba = rf_model.predict_proba(X_test)

        # Calculate the AUC score
        auc = roc_auc_score(y_test, y_pred_proba, average=None, multi_class='ovr')  # Assuming you're interested in the AUC for the positive class
        print("AUC:", auc)
        return list(auc)[::-1]

    def test(self, metrics=False):
        # Creates a SVC object with the best parameters selected.
        self.test_classifier = RandomForestClassifier(
            bootstrap = self.grid_search.best_params_['bootstrap'],
            n_estimators = self.grid_search.best_params_['n_estimators'],
            max_depth = self.grid_search.best_params_['max_depth'],
            class_weight = self.grid_search.best_params_['class_weight'],
            min_samples_split = self.grid_search.best_params_['min_samples_split'],
            max_features = self.grid_search.best_params_['max_features'],
            criterion = self.grid_search.best_params_['criterion'],
            random_state = 42 # Ensure reproducibility
        )

        # Runs a test to find the accuracy of the model
        self.test_classifier.fit(self.X_train, self.y_train)

        y_testresult = self.test_classifier.predict(self.X_test)

        # Now display the metrics if needed
        if metrics == True:
            super().display_class_results_text(y_testresult, self.y_test, self.get_auc(self.X_train, self.y_train, self.X_test, self.y_test))

    def relative_importance(self):
        # Get the importance of each feature in this classifier
        importances = self.test_classifier.feature_importances_

        X = self.df.drop("DX", axis=1)

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        # Plot the Graph
        plt.bar(X.columns.tolist(), importances)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.show()