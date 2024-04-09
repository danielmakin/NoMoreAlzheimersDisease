from Classifiers.MyMLP import MyMLP

def test(path, fields, file_output):
    mlp = MyMLP(path, fields)

    # Before Removing Outliers
    print("Computing Results.......")
    mlp.hyper_parameter_selection(verbose=0)
    mlp.test(metrics=True, max_iterations=2000)
    mlp.plot_loss(file_output)


print("----------Testing With Fields (File 1 (U)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT']")
test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'], 'mlp_output/f1u1')
print("----------Testing With Fields (File 1(U)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU']")
test('PreProcessingFiles/Data/PreProcessedData/File1/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU'], "mlp_output/f1u2")

print("----------Testing With Fields (File 1 (C)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT']")
test('PreProcessingFiles/Data/PreProcessedData/File1/CleanedData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT'], "mlp_output/f1c1")
print("----------Testing With Fields (File 1 (C)): ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU']")
test('PreProcessingFiles/Data/PreProcessedData/File1/CleanedData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMATAU', 'AB4240', 'MMSE', 'AGE', 'PTEDUCAT', 'ABETA','PTAU','TAU'], "mlp_output/f1c2")

print("----------Testing With Fields (File 2 (U)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER']")
test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER'], "mlp_output/f2u1")
print("----------Testing With Fields (File 2 (U)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER']")
test('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER'], "mlp_output/f2u2")

print("----------Testing With Fields (File 2 (C)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER']")
test('PreProcessingFiles/Data/PreProcessedData/File2/CleanedData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','ABETA','PTAU','TAU','MMSE','PTEDUCAT','AGE','PTGENDER'], "mlp_output/f2c1")
print("----------Testing With Fields (File 2 (C)): ['DX','PLASMA_NFL','PLAMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER']")
test('PreProcessingFiles/Data/PreProcessedData/File2/CleanedData/data.csv', ['DX','PLASMA_NFL','PLASMAPTAU181','AB4240','MMSE','PTEDUCAT','AGE','PTGENDER'], "mlp_output/f2c2")