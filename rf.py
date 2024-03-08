from Classifiers.MyRF import MyRF

def test(path, fields):
    rf = MyRF(path, fields)

    # Before Removing Outliers
    print("Computing Results.......")
    rf.hyper_parameter_selection(iterations=50)




def main():
    test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'])
    
main()