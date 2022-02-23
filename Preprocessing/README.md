```
raw_data/
├── csv/
│   ├── (respondent_id)_(sequence_id).csv
│   └── ...
├── parquet/
│   ├── (respondent_id)_(sequence_id).parquet
│   └── ...

pre_processed_data/
├── iMotionsToPython/ #list of dataframe with data sampled to 128Hz
│   ├── BySequence/
│   │   ├── (respondent_id)_(sequence_id)_full_features_data.array
│   │   ├── (respondent_id)_(sequence_id)_annotation_data
│   │   └── (respondent_id)_(sequence_id)_dict
│   │   └── ...
│   └── AllSquence/ #Aggregated file
│   │   └── all_data_full_features_10_classes.array
├── processed_sensor_combination/(fold_number)/
│   ├── Sensor n x Sensor m x Sensor k/
│   │   ├── X_train.npy
│   │   ├── Y_train.npy
│   │   ├── X_test.npy
│   │   └── Y_test.npy
│   ├── Sensor n x Sensor m/
│   │   └── ...
│   └── ...
```

![processing]( /rsc/file_processing.png "processing")

A colab notebook is available to read samples from the processed files. 
https://colab.research.google.com/drive/1EAeAQLlCV4_zUB1Wvhfn4D_5h84bawCx?usp=sharing

# pre_processing.py 

- Read raw CSV files
- **Resample** signals to 128Hz
- **Segment** each Self-Efficacy answer (SE from reading and from assembly task)
- **Export** processed data in a pickle array of multi-indexed dataframe
    - Each dataframe representing the signals from a SE answer 
```
pre_processed_data/
├── iMotionsToPython/ #list of dataframe with data sampled to 128Hz
│   ├── BySequence/
│   │   ├── (respondent_id)_(sequence_id)_full_features_data.array
│   │   ├── (respondent_id)_(sequence_id)_annotation_data
│   │   └── (respondent_id)_(sequence_id)_dict
│   │   └── ...
│   └── AllSquence/ #Aggregated file
│   │   └── all_data_full_features_10_classes.array
```

**To run only one CSV (sequence) file:**
```
export_path = './iMotionsToPython/'
filename="./raw_data/(respondent_id)_(sequence_id).csv"
sequence_list = []
sequence_list = runOneFile(filename,export_path, False, False, False)
```

From this function, we obtain by sequence :
- *(respondent_id)_(sequence_id)_annotation_data* : an annotation file with all the SE answer (raw_data without signals)
- *(respondent_id)_(sequence_id)_dict* :a dictionnary which contain for each assembly task: all the SE answer and the time to complete.
- *(respondent_id)_(sequence_id)_full_features_data.array* : list of dataframe segmented by SE with all the data resampled to 128Hz.

**To run all CSV files in a repository:**
```
export_path = './iMotionsToPython/AllSequence/'
path = './raw_data/'
runAllFilesInRepository(path, export_path)
```
This fucntion call the runOneFile function hence obtaining all the files aboves by sequence and an aggregated file of the *(respondent_id)_(sequence_id)_full_features_data.array* : 
- *all_data_full_features_10_classes.array* 

# run_experiment.py

**Prepare train and test files on all combinaison of sensor available.**

input : *_full_features_data.array*

- Read  data exported by pre_processing.py

```
AllData = pd.read_pickle("pre_processed/iMotionsToPython/Allsequence/all_data_full_features_10_classes_final.array")
```
- **Create cross-validation fold**
- Data slicing :
  - With or w/o Data augmentation by **window overlapping**



![window]( /rsc/window_slicing.png
 "window")

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

**To run the file to make all combination of groups with an n folds for each combination**

```
crossValSplit(folds_number, AllData, groups, folder_name_prefix= "_data_with_augmentation_byFeatures")```

