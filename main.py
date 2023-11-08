import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    # read the dataframe and clean
    SCD, MCI, AD = clean_data('CSFplasma.csv')
    # get the test data required, leave training data
    SCD, MCI, AD, TestData = removeTestingData(SCD, MCI, AD)
    # now make each svm

    

    #IDEA
        #make a function for testing in general
        #split data before creating 3 SVMs
        #have objects for each of the three SVMs
        #perform one against one approach
        #result
    

    

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
    MCI = df.loc[df["DX"] == "MCI"].drop_duplicates(subset= "RID")
    SCD = df.loc[df["DX"] == "SCD"].drop_duplicates(subset= "RID")
    AD = df.loc[df["DX"] == "AD"].drop_duplicates(subset= "RID")

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
    SCD, TempSCD = train_test_split(SCD, test_size=TestingFactor, random_state=0)
    MCI, TempMCI = train_test_split(MCI, test_size=TestingFactor, random_state=0)
    AD, TempAD = train_test_split(AD, test_size=TestingFactor, random_state=0)

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
            Classifier (SVC) : This is the margin that the data must be acted upon'''
    
    
    
    # First the two dataframes should be combined
    DataSet = np.concat([DataSet1, DataSet2])
    # Get data in the format required
    X, y = getXy(DataSet)
    # Train the classifier
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # Fit to the classifier
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X, y)

    return classifier
    

    
main()