'''This is a Pre-Processing Python file to make the PreProcessing.ipynb file cleaner'''
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

class pp:
    def __init__(self, files, merge_fields, fields_needed):
        ## MAY HAVE ISSUE IF THERE IS VISCODE/2 ETC.

        # Read the first file
        self.df = pd.read_csv(files[0])

        # Merge in the other files, ignoring the original as it has already been read
        for i in range(1, len(files)):
            temp_df = pd.read_csv(files[i])
            self.df = pd.merge(self.df, temp_df, on=merge_fields, how="inner")
        # Select the Fields that are needed
        self.df = self.df[fields_needed]

    def getXy(self, df):
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

    def use_SMOTE(self):
        sm = SMOTE(random_state=42)

        X, y = self.getXy(self.df.drop(columns=['RID', 'VISCODE']))

        # Oversample the minority classes
        X, y = sm.fit_resample(X, y)

        self.df = pd.DataFrame(data=np.column_stack((X, y)), columns=list(self.df.drop(columns=["VISCODE", "RID", "DX"], axis=1).columns) + ["DX"])
    
    def create_ab4240(self, columns):
        '''Columns Should be Given in the order: AB42, AB40'''
        # Creates a column for the AB42/AB40 ratio
        self.df['AB4240'] = self.df[columns[0]] / self.df[columns[1]]
        self.df = self.df.drop([columns[0], columns[1]], axis=1)

    def add_adni_merge_data(self, fields_to_keep):
        '''This adds the data from the ADNI-MERGE file. i.e. DX, MMSE, AGE, PT_EDUCAT'''
        am = pd.read_csv('Data/ADNIMERGE_15Jun2023.csv')

        self.df = pd.merge(self.df, am, on=['RID', 'VISCODE'], how='inner')[fields_to_keep]


    def clean_data(self):
        '''Removes the NULL values and Makes DX a Manageable Name'''
        # The class Data should be in the Forms: SCD, MCI, AD
        self.df = self.df.replace("CN", "SCD").replace("Dementia", "AD").dropna()

    def remove_outliers(self, columns_to_clean):

        '''This removes the outliers from the specified Columns, returns the amount of records removed'''
        SCD = self.df.loc[self.df["DX"] == "SCD"].reindex()
        MCI = self.df.loc[self.df["DX"] == "MCI"].reindex()
        AD = self.df.loc[self.df["DX"] == "AD"].reindex()

        # Initialise to_remove

        to_remove = self.__remove_outliers_class(SCD, columns_to_clean)
        print(to_remove)
        SCD.drop(to_remove, inplace=True)
        to_remove = self.__remove_outliers_class(MCI, columns_to_clean)
        MCI.drop(to_remove, inplace=True)
        to_remove = self.__remove_outliers_class(AD, columns_to_clean)
        AD.drop(to_remove, inplace=True)

        # Now Recombine and reindex to ensure uniqueness
        df = [SCD, MCI, AD]
        self.df = pd.concat(df).reindex()
    
    def __remove_outliers_class(self, df, columns_to_clean):
        to_remove = []
        for i in range(len(columns_to_clean)):
            p25 = df[columns_to_clean[i]].quantile(0.25)
            p75 = df[columns_to_clean[i]].quantile(0.75)

            iqr = p75 - p25

            #Get the boundaries
            upper_limit = p75 + 1.5 * iqr
            lower_limit = p25 - 1.5 * iqr
            
            u = df[df[columns_to_clean[i]] > upper_limit]
            l = df[df[columns_to_clean[i]] < lower_limit]

            # Making it a set removes duplicates
            to_remove = to_remove + (list(u.index) + list(l.index))

        # Now make it a set to remove duplicates
        to_remove = list(set(to_remove))

        return to_remove
    
    def write_to_csv(self, file_name):
        '''Writes a dataframe to the specified .csv file'''
        # Assume this is a pre-processing file
        self.df.to_csv("Data/PreProcessedData/" + file_name, index=False)

    
class visual_display:

    def __init__(self, df):
        self.SCD = df.loc[df["DX"] == "SCD"]
        self.MCI = df.loc[df["DX"] == "MCI"]
        self.AD = df.loc[df["DX"] == "AD"]

    def display(self, fields):
        fig, axes = plt.subplots(nrows=1, ncols=len(fields), figsize=(20, 4))

        for i in range(len(fields)):
            box_plot = [list(self.SCD[fields[i]]), list(self.MCI[fields[i]]), list(self.AD[fields[i]])]

            axes[i].boxplot(box_plot, showfliers=True, labels=['SCD', 'MCI', 'AD'])
            axes[i].set_title(fields[i] + ' values on BoxPlots')
            axes[i].set_ylabel(fields[i] + ' Value')
            axes[i].set_xlabel('Classification')

        plt.show()


class post_processing_display:

    def __init__(self, file_name):
        self.before_dfs = []
        self.after_dfs = []
        # This should read all of the data frames needed

        self.before = pd.read_csv('Data/PreProcessedData/' + file_name + "/UnCleanData/data.csv")
        self.filtered = pd.read_csv('Data/PreProcessedData/' + file_name + "/CleanedData/data.csv")
        self.smote = pd.read_csv('Data/PreProcessedData/' + file_name + "/SMOTEData/data.csv")

    def display_results(self):
        plt.figure(figsize=(8,5))
        # Split into arrays where each item has a list from a class from a file
        SCD, MCI, AD = [], [], []

        SCD.append(self.before.loc[self.before["DX"] == "SCD"])
        SCD.append(self.filtered.loc[self.filtered["DX"] == "SCD"])
        SCD.append(self.smote.loc[self.smote["DX"] == "SCD"])

        MCI.append(self.before.loc[self.before["DX"] == "MCI"])
        MCI.append(self.filtered.loc[self.filtered["DX"] == "MCI"])
        MCI.append(self.smote.loc[self.smote["DX"] == "MCI"])

        AD.append(self.before.loc[self.before["DX"] == "AD"])
        AD.append(self.filtered.loc[self.filtered["DX"] == "AD"])
        AD.append(self.smote.loc[self.smote["DX"] == "AD"])

        # Now display what this means

        r = np.arange(3)
        width = 0.25
        plt.bar(r-width, [len(SCD[0]), len(MCI[0]), len(AD[0])], label='Before', width=width, edgecolor = 'black')
        plt.bar(r, [len(SCD[1]), len(MCI[1]), len(AD[1])], label='Filtered', width=width, edgecolor = 'black')
        plt.bar(r+width, [len(SCD[2]), len(MCI[2]), len(AD[2])], label='SMOTE', width=width, edgecolor = 'black')
        plt.title("Distribution of Classes")
        plt.ylabel('Class Size')
        plt.xticks(r, ['SCD','MCI','AD'])
        plt.xlabel('Classification')
            
        plt.legend()
        plt.show()


    