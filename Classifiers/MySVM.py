import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

from Classifiers.MyClassifier import MyClassifier


class MySVM(MyClassifier):
    def __init__(self, file_name, fields):
        # Get the .csv and remove the unnecessary fields
        self.df = pd.read_csv(file_name)[fields]
        # Now we can remove the null values
        self.df.dropna(inplace=True)
    
    def hyper_parameter_selection(self, verbose=0, file_name=""):
        self.svm = SVC()
        X, y = super().getXy(self.df)

        # Split the data, this may need to be fixed later, the Test data will be needed after
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # These are the parameters that will be trialled
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False],
            'probability': [True, False],
            'class_weight': ['balanced'],
            'tol': [1e-3, 1e-4, 1e-5],
            'decision_function_shape': ['ovo', 'ovr']
}

        self.parameters = ['C', 'kernel', 'degree', 'decision_function_shape', 'shrinking', 'coef0', 'probability', 'class_weight', 'tol']

        # Runs every possible combination and gets the best
        grid_search = RandomizedSearchCV(estimator=self.svm, param_distributions=self.param_grid, cv=3, scoring='accuracy', verbose=verbose, n_iter=50, n_jobs=-1)

        grid_search.fit(self.X_train, self.y_train)

        # Display the best parameters and best accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        # Store these values for next function

        self.grid_search = grid_search

        # Now Display in the Notebook the output of these results

        super().display_hyperparameter_results(grid_search, self.parameters, self.param_grid, file_name)
        # super().display_results_against_iterations(self.grid_search.cv_results_)

    def test(self, metrics=False):
        
        # Creates a SVC object with the best parameters selected.
        test_classifier = SVC(
            C = self.grid_search.best_params_['C'],
            decision_function_shape = self.grid_search.best_params_['decision_function_shape'],
            degree = self.grid_search.best_params_['degree'],
            kernel = self.grid_search.best_params_['kernel'],
            coef0 = self.grid_search.best_params_['coef0'],
            probability = self.grid_search.best_params_['probability'],
            # gamma = self.grid_search.best_params_['gamma'],
            shrinking = self.grid_search.best_params_['shrinking'],
            tol = self.grid_search.best_params_['tol'],
            class_weight = self.grid_search.best_params_['class_weight']
        )

        # Runs a test to find the accuracy of the model
        test_classifier.fit(self.X_train, self.y_train)

        y_testresult = test_classifier.predict(self.X_test)

        # Now display the metrics if needed
        if metrics == True:
            super().display_class_results_text(y_testresult, self.y_test, super().auc_scores_svm(self.X_train, self.y_train, self.X_test, self.y_test))