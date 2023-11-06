#this script can be used to extract the important data from the .csv file
#also clean any non alpha numeric data from the fields

import pandas as pd
import numpy as np

def main(fileName):


    df = pd.read_csv(fileName)

    #baseline values are not needed as there is an initial entry already
    df = df.loc[:, ~df.columns.str.endswith('_bl')]

    #get the relevant biomarkers
    df = df.loc[:, ['RID', 'DX', 'MMSE', 'AGE', 'PTAU', 'TAU', 'ABETA']]
    df = df.replace(">1700", "1700")

    return df


