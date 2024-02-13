'''This is a Pre-Processing Python file to make the PreProcessing.ipynb file cleaner'''
import pandas as pd

class pp:
    def __init__(self, files, merge_fields, fields_needed):
        # Read the first file
        self.df = pd.read_csv(files[0])

        # Merge in the other files, ignoring the original as it has already been read
        for i in range(1, len(files)):
            temp_df = pd.read_csv(files[i])
            self.df = pd.merge(self.df, temp_df, on=merge_fields, how="inner")#
        # Select the Fields that are needed
        self.df = self.df[fields_needed]
    
    def create_ab4240(self, columns):
        '''Columns Should be Given in the order: AB42, AB40'''
        # Creates a column for the AB42/AB40 ratio
        self.df['AB4240'] = self.df[columns[0]] / self.df[columns[1]]
        self.df = self.df.drop([columns[0], columns[1]], axis=1)

    def add_adni_merge_data(self):
        '''This adds the data from the ADNI-MERGE file. i.e. DX, MMSE, AGE, PT_EDUCAT'''
        am = pd.read_csv('Data/ADNIMERGE_15Jun2023.csv')
        self.df = pd.merge(self.df, am, on=['RID', 'VISCODE'], how='inner')[['RID', 'VISCODE', 'DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'PTEDUCAT', 'AGE']]

    def clean_data(self):
        '''Removes the NULL values and Makes DX a Manageable Name'''
        # The class Data should be in the Forms: SCD, MCI, AD
        self.df = self.df.replace("CN", "SCD").replace("Dementia", "AD").dropna()

    def remove_outliers(self, columns_to_clean):
        '''This removes the outliers from the specified Columns, returns the amount of records removed'''
        to_remove = []

        for field in range(columns_to_clean):
            p25 = self.df[field].quantile(0.25)
            p75 = self.df[field].quantile(0.75)

            iqr = p75 - p25

            #Get the boundaries
            upper_limit = p75 + 1.5 * iqr
            lower_limit = p25 - 1.5 * iqr
            
            u = self.df[self.df[field] > upper_limit]
            l = self.df[self.df[field] < lower_limit]

            # Making it a set removes duplicates
            to_remove = to_remove + (list(u['RID']) + list(l['RID']))
        # Now make it a set to remove duplicates
        to_remove = list(set(to_remove))
        return len(to_remove)
             