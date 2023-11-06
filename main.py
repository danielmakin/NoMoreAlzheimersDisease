import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    SCDandMCI, MCIandAD, SCDandAD = clean_data('CSFplasma.csv')
    construct_svm(MCIandAD)

    #IDEA
        #make a function for testing in general
        #split data before creating 3 SVMs
        #have objects for each of the three SVMs
        #perform one against all approach
        #result
    

    

def clean_data(fileName):
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

    SCDandMCI = [SCD, MCI]
    MCIandAD = [MCI, AD]
    SCDandAD = [SCD, AD]
    
    #concatenate for SVM margins
    SCDMCI = pd.concat(SCDandMCI)
    MCIAD = pd.concat(MCIandAD)
    SCDAD = pd.concat(SCDandAD)

    return SCDMCI, MCIAD, SCDAD

def construct_svm(dataset):
    #independant
    X = dataset.iloc[:, [2, 3, 4, 5, 6]].values
    #dependant
    y = dataset.iloc[:, 1].values
    
    #split into testing and training sets (0.25 for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    #fit the SVM to the training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    #test the classifier
    y_pred = classifier.predict(X_test)

    #construct a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    
main()