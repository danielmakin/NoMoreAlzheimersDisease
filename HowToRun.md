# How To Run

## Necessary Modules
* sklearn
* pandas
* numpy
* matplotlib.pyplot
* imblearn
## File Explanations
* Files:
    * `rf.py`, `svm.py` and `mlp.py`: These are the Experimental Files for Each Respective Classifier.
        * These can be Ran Using **python3 <classifier>** (For the Sake of testing these have been outputted to a file.)
        * These Have been Outputted to `mlp_output/`, `rf_output/` and `mlp_output/` Respectively.
    * `PreProcessing.ipynb`, `PreProcessing2.ipynb`: These are the Respective Data Pre-Processing Files.
        * Located in `PreProcessingFiles/`
        * `PreProcessingFiles/PreProcecssingPy.py` Contains the Logic used for these Jupyter Notebooks.
    * `PreProcessingFiles/Data`: This Houses any PreProcessing (Actual Raw Data), This includes Method Files that May have been used for Analysis.

* Classifiers:
    * This Module is Used for All Classification Issues and Relies upon the `sklearn` library.
    * `MyClassifier.py` is the Base class that Also houses Some Useful Modules used by all.
    * `MyMLP`, `MyRF`, `MySVM` Have each been derived from `MyClassifier.py`.
