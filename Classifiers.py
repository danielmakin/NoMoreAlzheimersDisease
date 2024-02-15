from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

class MySVM:
    def __init__(self, file_name, fields_to_drop):
        self.df = pd.read_csv(file_name).drop(fields_to_drop, axis=1)
    
    def hyper_parameter_selection(self, verbose=0):
        self.svm = SVC()
        X, y = self.__getXy(self.df)

        # Split the data, this may need to be fixed later, the Test data will be needed after
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # These are the parameters that will be trialled
        self.param_grid = {
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel' : ['linear', 'poly', 'rbf'],
            'degree' : [2, 3, 4],
            'decision_function_shape' : ['ovr', 'ovo']
        }

        self.parameters = ['C', 'kernel', 'degree', 'decision_function_shape']

        # Runs every possible combination and gets the best
        grid_search = GridSearchCV(estimator=self.svm, param_grid=self.param_grid, cv=3, scoring='accuracy', verbose=verbose)

        grid_search.fit(self.X_train, self.y_train)

        # Display the best parameters and best accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        # Store these values for next function

        self.grid_search = grid_search

    def __getXy(self, df):
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
    
    def display_hyperparameter_results(self):
        display = main_results_display()

        display.display_hyperparameter_results(self.grid_search, self.parameters, self.param_grid)

    def test_hyper_parameters(self):
        
        # Creates a SVC object with the best parameters selected.
        test_classifier = SVC(
            C = self.grid_search.best_params_['C'],
            decision_function_shape = self.grid_search.best_params_['decision_function_shape'],
            degree = self.grid_search.best_params_['degree'],
            kernel = self.grid_search.best_params_['kernel']
        )

        # Runs a test to find the accuracy of the model
        test_classifier.fit(self.X_train, self.y_train)

        y_testresult = test_classifier.predict(self.X_test)

        print("Accuracy is " + str(accuracy_score(self.y_test, y_testresult)))
        print("Recall was " + str(recall_score(self.y_test, y_testresult, average=None)))

class MyRF:
    def __init__(self, file_name, fields_to_drop):
        self.df = pd.read_csv(file_name).drop(fields_to_drop, axis=1)

    def __getXy(self, df):
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

    def hyper_parameter_selection(self, verbose=0):
        # Create the test object
        self.rf = RandomForestClassifier()

        X, y = self.__getXy(self.df)

        # Split for a 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Now we create the grid of params we need to test
        param_grid = {
            'n_estimators' : [50, 100, 150, 200, 250, 300],
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'bootstrap' : [True, False],
            'class_weight' : ['balanced', 'balanced_subsample']
        }

        # Create the test Object and fit
        grid_search = GridSearchCV(estimator=self.rf, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        self.parameters = ['n_estimators', 'criterion', 'bootstrap', 'class_weight']
        # Display the best parameters and best accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        # Store these values for next function

        self.grid_search = grid_search

    def test_hyper_parameters(self):
        
        # Creates a SVC object with the best parameters selected.
        test_classifier = RandomForestClassifier(
            n_estimators = self.grid_search.best_params_['n_estimators'],
            criterion = self.grid_search.best_params_['criterion'],
            bootstrap = self.grid_search.best_params_['bootstrap'],
            class_weight = self.grid_search.best_params_['class_weight']
        )

        # Runs a test to find the accuracy of the model
        test_classifier.fit(self.X_train, self.y_train)

        y_testresult = test_classifier.predict(self.X_test)

        print("Accuracy is " + str(accuracy_score(self.y_test, y_testresult)))
        print("Recall was " + str(recall_score(self.y_test, y_testresult, average=None)))

class main_results_display:
    '''This Class will be use by both the SVM and RF approach'''
    def __init__(self):
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

# TODO :: Comment and fix the fact that there is no x label for top grpahs
# TODO :: See if the loss function cab be printed