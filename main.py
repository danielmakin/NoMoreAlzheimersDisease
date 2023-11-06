import pandas as pd
import numpy as np

def main():
    SCDandMCI, MCIandAD, SCDandAD = clean_data('CSFplasma.csv')
    print(SCDandAD)
    

    

def clean_data(fileName):


    df = pd.read_csv(fileName)

    #baseline values are not needed as there is an initial entry already
    df = df.loc[:, ~df.columns.str.endswith('_bl')]

    #get the relevant biomarkers
    df = df.loc[:, ['RID', 'DX', 'MMSE', 'AGE', 'PTAU', 'TAU', 'ABETA']]
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

main()