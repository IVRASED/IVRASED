**pre_processing.py - Read Imotion CSV files, and make all data on the same sample rate (128) - Export new data in an array (pickle)**

**To run only one Imotion csv file:**
```
export_path = '/home/deep01/zaher/new_export/exportedObjectFullFeatures/'
filename="/home/deep01/imotions_data/003_3754b5_sequence2.csv"
sequence_list = []
sequence_list = runOneFile(filename,export_path, False, False, False)
```

**To run all Imotion csv files in a repository:**
```
export_path = '/home/deep01/zaher/new_export/exportedObjectFullFeatures/'
path = '/home/deep01/imotions_data_juin'
runAllFilesInRepository(path, export_path)
```

**run_experiment.py - Prepare data to deep architecture**

**Step 1 - read data exported by pre_processing.py**

`
AllData = pd.read_pickle("/home/deep01/zaher/all_data_full_features_10_classes_final.array")
`

**Step 2 - add your parameters for the data to export**

**Example of parameters to export data(train and test) without overlap - take only last 2 seconds of each SEF**

```
window_size_in_sec=2
data_to_slice_full_time=2
slice_size=0.5
sample_rate=128
data_folder_name = ""
folds_number = 5
```

**Example of parameters to export data(train and test) with overlap - For each SEF, take last 10 seconds and compose it to a set of windows of 2 seconds each with 0,5 of overlap between each window**

```
window_size_in_sec=2
data_to_slice_full_time=10
slice_size=0.5
sample_rate=128
data_folder_name = ""
folds_number = 5
```

**Step 3 - Run the file to make all combination of groups with an n folds for each combination**

`crossValSplit(folds_number, AllData, groups, folder_name_prefix= "_data_with_augmentation_2C_byFeatures_all_new")`

