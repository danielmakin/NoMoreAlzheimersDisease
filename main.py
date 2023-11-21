import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    # read the dataframe and clean
    SCD, MCI, AD = clean_data('Data/CSFplasma.csv')
    # get the test data required, leave training data
    SCD, MCI, AD, TestData = removeTestingData(SCD, MCI, AD)
    # perform the testing
    doTesting(SCD, MCI, AD, TestData)


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

    #remove duplicates in each subset
    MCI = df.loc[df["DX"] == "MCI"]#.drop_duplicates(subset= "RID")
    SCD = df.loc[df["DX"] == "SCD"]#.drop_duplicates(subset= "RID")
    AD = df.loc[df["DX"] == "AD"]#.drop_duplicates(subset= "RID")

    #we need the same of every class to simply select the first ones
    # sizes = [len(SCD), len(MCI), len(AD)]
    # minAmount = min(sizes)
    # print(minAmount)
    #
    # SCD = SCD[:minAmount]
    # MCI = MCI[:minAmount]
    # AD = AD[:minAmount]

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

def doTesting(SCD, MCI, AD, TestData):
    '''
        Make the individual SVMs and test which class it belongs to using majority voting

        Args :
            SCD, MCI, AD (DataFrames) : The inputted dataframes for training
            TestData (DataFrame) : The dataframe used for testing
    '''
    # make the individual SVMs and keep the scalers for testing

    X_test, y_test = getXy(TestData)

    SCDMCI, scalersm = construct_svm(SCD, MCI)
    MCIAD, scalerma = construct_svm(MCI, AD)
    SCDAD, scalersa = construct_svm(SCD, AD)

    #perform each test
    test1 = test(SCDMCI, scalersm, X_test)
    test2 = test(MCIAD, scalerma, X_test)
    test3 = test(SCDAD, scalersa, X_test)

    # now to get the most common item in each index, leaving an item as unclassified if need be

    commonResults = getMostCommonResult(test1, test2, test3)

    # construct a confusion matrix
    cm = confusion_matrix(y_test, commonResults)
    print(cm)
    print(accuracy_score(y_test, commonResults))

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
