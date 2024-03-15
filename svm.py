from Classifiers.MySVM import MySVM

def test(path, fields, file_output):
    svm = MySVM(path, fields)

    # Before Removing Outliers
    print("Computing Results.......")
    svm.hyper_parameter_selection(file_name=file_output, verbose=3)
    svm.test(metrics=True)




def main():
    print("----------Testing With Fields (File 1 (U)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT']")
    test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'], "svm_output/f1u1")
    print("----------Testing With Fields (File 1(U)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU']")
    test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU'], "svm_output/f1u2")

    print("----------Testing With Fields (File 1 (C)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT']")
    test('PreProcessingFiles/Data/PreProcessedData/File1/CleanedData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'], "svm_output/f1c1")
    print("----------Testing With Fields (File 1 (C)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU']")
    test('PreProcessingFiles/Data/PreProcessedData/File1/CleanedData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU'], "svm_output/f1c2")

    print("----------Testing With Fields (File 2 (U)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER']")
    test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER'], "svm_output/f2u1")
    print("----------Testing With Fields (File 2 (U)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER']")
    test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER'], "svm_output/f2u2")

    print("----------Testing With Fields (File 2 (C)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER']")
    test('PreProcessingFiles/Data/PreProcessedData/File2/CleanedData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER'], "svm_output/f2c1")
    print("----------Testing With Fields (File 2 (C)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER']")
    test('PreProcessingFiles/Data/PreProcessedData/File2/CleanedData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER'], "svm_output/f2c2")
    # test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240', 'TAU','MMSE','PTEDUCAT','AGE','PTGENDER'])
    
main()