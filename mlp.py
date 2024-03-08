from Classifiers.MyMLP import MyMLP

print("-----TEST 1--------")
mlp = MyMLP('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMAPTAU181', 'AB4240', 'MMSE', 'PTEDUCAT', 'AGE'])
mlp.test(metrics=True)
print("----TEST 2---------")
mlp = MyMLP('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMAPTAU181', 'AB4240'])
mlp.test(metrics=True)
print("----TEST 3--------")
mlp = MyMLP('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMAPTAU181', 'AB4240', 'MMSE', 'PTEDUCAT', 'AGE', 'ABETA', 'PTAU', 'TAU'])
mlp.test(metrics=True)
print("----TEST 4---------")
mlp = MyMLP('PreProcessingFiles/Data/PreProcessedData/File2/UnCleanData/data.csv', ['DX', 'PLASMA_NFL', 'PLASMAPTAU181', 'MMSE', 'AGE', 'PTEDUCAT'])
mlp.test(metrics=True)