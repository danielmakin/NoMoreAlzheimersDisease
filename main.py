''''Approach :  Make a Binary DT Classifier which uses mostly the same logic as the last iteration
                This requires preprocessing the classes to make two the same for the first classifier
                The Current idea :
                            ALL
                           /   \
                          /     \
                        SCD   MCI,AD
                                /\
                               /  \
                             MCI  AD
                '''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from timeit import default_timer as timer

def main():
    start = timer()
    # read the dataframe and clean
    SCD, MCI, AD = clean_data('Data/CSFplasma.csv')
    # get the test data required, leave training data
    SCD, MCI, AD, TestData = removeTestingData(SCD, MCI, AD)
    # perform the testing
    doTesting(SCD, MCI, AD, TestData)

    # Print out the time taken
    end = timer()
    print("Time Taken : " + str(end-start))


def clean_data(fileName):
    '''
        Reads a .cvs file and returns the SCD, MCI, and AD associated data.

        Args :
            fileName (String): name of the .csv file (including extension)
        Returns :
            SCD (list) : List of SCD patients and associated data
            MCI (list) : List of MCI patients and associated data
            AD (list) : List of AD patients and associated data
    '''


    df = pd.read_csv(fileName)

    #baseline values are not needed as there is an initial entry already
    df = df.loc[:, ~df.columns.str.endswith('_bl')]

    #get the relevant biomarkers
    df = df.loc[:, ['RID', 'DX', 'MMSE', 'AGE', 'PTAU', 'TAU', 'ABETA']]

    df = df.replace("<8", "8")
    df = df.replace(">1700", "1700")

    #change to required format
    df = df.replace("CN", "SCD")
    df = df.replace("Dementia", "AD")

    #get items from each class
    MCI = df.loc[df["DX"] == "MCI"]
    SCD = df.loc[df["DX"] == "SCD"]
    AD = df.loc[df["DX"] == "AD"]

    return SCD, MCI, AD

def getXy(dataset):
    '''
        Returns the X (features) and y (label) associated with the dataframe

        Args :
            dataset (dataframe)
        Returns :
            X (dataframe) : features
            y (list) : labels
    '''
    #independant
    X = dataset.iloc[:, [2, 3, 4, 5, 6]].values
    #dependant
    y = dataset.iloc[:, 1].values

    return X, y

def removeTestingData(SCD, MCI, AD, TestingFactor = 0.25):
    '''
        Function to remove the testing data from the DataFrames.

        Args :
            SCD, MCI, AD (DataFrame): Dataframes associated with each condition
            TestingFactor (float) : Amount of data held back for testing (default = 0.25)
        Returns :
            SCD, MCI, AD (DataFrame): Dataframes with the testing data removed
            Test (DataFrame): Testing Data
    '''

    # remove the non-allowed data
    SCD, TempSCD = train_test_split(SCD, test_size=TestingFactor)
    MCI, TempMCI = train_test_split(MCI, test_size=TestingFactor)
    AD, TempAD = train_test_split(AD, test_size=TestingFactor)

    # concatenate the lists
    TempData = [TempSCD, TempMCI, TempAD]
    Test = pd.concat(TempData)

    # return required info
    return SCD, MCI, AD, Test

def construct_svm(DataSet1, DataSet2):
    '''
        Constructs an SVM with the datasets provided.

        Args :
            Datasets (DataFrame) : Two DataFrames that the SVM must be made upon
        Returns :
            Classifier (SVC) : This is the margin that the data must be acted upon
    '''



    # First the two dataframes should be combined
    DataSet = [DataSet1, DataSet2]
    DataSet = pd.concat(DataSet)
    # Get data in the format required
    X, y = getXy(DataSet)
    # Train the classifier
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # Fit to the classifier
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X, y)

    return classifier, sc

def perform_bdt(SCDoMCIAD, MCIoAD):
    '''This collates the two levels of the BDT to produce a single class for each data item
    
    Args : 
        SCDoMCIAD, MCIoAD (List) : The inputted dataframes to collate.
    Returns :
        Results (List): The full result of the BDT'''
    
    Results = []
    
    for i in range(len(SCDoMCIAD)):
        if SCDoMCIAD[i] == 'SCD':
            Results.append('SCD')
        else:
            # Adds the result of the second level
            Results.append(MCIoAD[i])

    return Results

def doTesting(SCD, MCI, AD, TestData):
    '''
        Make the individual SVMs and test which class it belongs to using Binary Decision Tree Classification

        Args :
            SCD, MCI, AD (DataFrames) : The inputted dataframes for training
            TestData (DataFrame) : The dataframe used for testing
    '''
    # seperate the labels and the data features
    X_test, y_test = getXy(TestData)

    # Concatenate MCI
    MCIoAD = [MCI, AD]
    MCIoAD = pd.concat(MCIoAD)
    # Map to the same class
    MCIoAD = MCIoAD.replace("MCI", "MCIoAD").replace("AD", "MCIoAD")

    # Test to perform the first step of the DT
    SCDMCIAD, scalersma = construct_svm(SCD, MCIoAD)
    # SVM for the second level of the BDT
    MCIAD, scalarma = construct_svm(MCI, AD)

    # Perform the test for the first level of BDT
    test1 = test(SCDMCIAD, scalersma, X_test)

    # Now run entire set through the MCI, AD classifier. 
    # However, only the non-SCD items in the previous test will be used
    test2 = test(MCIAD, scalarma, X_test)

    # Now colalate the results together taking the two classifiers into account
    results = perform_bdt(test1, test2)

    # construct a confusion matrix
    cm = confusion_matrix(y_test, results)
    print(cm)
    print(accuracy_score(y_test, results))

def getMostCommonResult(pred1, pred2, pred3):
    '''
        This allows the collation of all the results

        Args :
            pred1, pred2, pred3 (lists) : This gives the prediction from each individual SVM
        Returns :
            common (list) : This returns the most common classification
    '''
    common = ["-1" for i in range(len(pred1))]
    notClassified = 0

    # classify values that have atleast a 2 in a majority voting scheme
    for i in range(len(pred1)):
        if (pred1[i] == pred2[i]) or (pred1[i] == pred3[i]):
            common[i] = pred1[i]
        elif (pred2[i] == pred3[i]):
            common[i] = pred2[i]
        else:
            notClassified += 1

    # lets the user know if any items cannot be classified.
    if notClassified != 0:
        print("There were " + str(notClassified) + " item(s) unclassified.")
    return common

def test(classifier, scaler, X):
    '''
        Test the dataset with each individual SVM

        Args :
            classifier (SVM) : The Support Vector Machine used for this test
            scaler (StandardScaler) : This allows the test data to be scaled to the same proportions as the test data
            X (DataFrame) : The feature data WITHOUT labels
        Returns :
            y_pred (list) : the predicted y-value for each item
    '''
    # now perform the classification
    X = scaler.fit_transform(X)
    y_pred = classifier.predict(X)
    # return the result of this transcation
    return y_pred

main()
