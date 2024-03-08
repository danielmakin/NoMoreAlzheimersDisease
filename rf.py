from Classifiers.MyRF import MyRF

def test(path, fields):
    rf = MyRF(path, fields)

    # Before Removing Outliers
    print("Computing Results.......")
    rf.hyper_parameter_selection(iterations=50)
    rf.test(metrics=True)




def main():
    # test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'])
    test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240', 'TAU','MMSE','PTEDUCAT','AGE','PTGENDER'])
    
main()