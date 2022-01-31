import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import seaborn as sns
import matplotlib
import matplotlib.animation as animation

#from datetime import timedelta
from hhpy.plotting import animplot

import datetime
from pandas import datetime as dt
from matplotlib import pyplot

import os, fnmatch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import itertools
import pickle


# List of groups of columns for EEG, ECG and ET - They are used to run experiments for combination between groups
eeg_current_columns1 = ['EEG_Decon_Poz', 'EEG_Decon_P3','EEG_Decon_P4']
eeg_current_columns2 = ['EEG_Decon_Cz', 'EEG_Decon_C3', 'EEG_Decon_C4']
eeg_current_columns3 = ['EEG_Decon_Fz', 'EEG_Decon_F3', 'EEG_Decon_F4']

ecg_columns1 = ['Other_Vsensebatt_RAW_Shim_Exg_02', 'Other_EXG1_Status_RAW_Shim_Exg_02', 'Other_EXG2_Status_RAW_Shim_Exg_02', 'Other_ECG_LL-RA_RAW_Shim_Exg_02', 'Other_ECG_LA-RA_RAW_Shim_Exg_02', 'Other_ECG_Vx-RL_RAW_Shim_Exg_02']
ecg_columns2 = ['Other_Vsensebatt_CAL_Shim_Exg_02', 'Other_ECG_LL-RA_CAL_Shim_Exg_02', 'Other_ECG_LA-RA_CAL_Shim_Exg_02', 'Other_ECG_Vx-RL_CAL_Shim_Exg_02']
ecg_columns3 = ['Other_Heart_Rate_ECG_LL-RA_ALG_Shim_Exg_02', 'Other_IBI_ECG_LL-RA_ALG_Shim_Exg_02']


#et_columns1 = []
et_columns1 = ['ET_Gaze_2D_ET_Gazeleftx', 'ET_Gaze_2D_ET_Gazelefty', 'ET_Gaze_2D_ET_Gazerightx', 'ET_Gaze_2D_ET_Gazerighty', 'ET_Capture_ET_Cameraleftx', 'ET_Capture_ET_Cameralefty', 'ET_Capture_ET_Camerarightx', 'ET_Capture_ET_Camerarighty', 'ET_Head_ET_Headrotationx', 'ET_Head_ET_Headrotationy', 'ET_Head_ET_Headrotationz', 'ET_Head_ET_Headpositionvectorx', 'ET_Head_ET_Headpositionvectory', 'ET_Head_ET_Headpositionvectorz','ET_Head_ET_Headvelocityx', 'ET_Head_ET_Headvelocityy', 'ET_Head_ET_Headvelocityz','ET_Head_ET_Headangularvelocityx', 'ET_Head_ET_Headangularvelocityy','ET_Head_ET_Headangularvelocityz', 'ET_Other_ET_Timesignal', 'ET_Distance_ET_Distanceleft','ET_Distance_ET_Distanceright']
et_columns2 = ['ET_Pupil_ET_Pupilleft', 'ET_Expression_ET_Lefteyeopenness''ET_Expression_ET_Lefteyesqueeze','ET_Expression_ET_Lefteyefrown']
et_columns3 = ['ET_Pupil_ET_Pupilright', 'ET_Expression_ET_Righteyesqueeze', 'ET_Expression_ET_Righteyeopenness', 'ET_Expression_ET_Righteyefrown']
              

all_columns_groups = {'eeg_current_columns1':eeg_current_columns1, 'eeg_current_columns2':eeg_current_columns2, 'eeg_current_columns3':eeg_current_columns3, 'ecg_columns1':ecg_columns1, 'ecg_columns2':ecg_columns2, 'ecg_columns3':ecg_columns3, 'et_columns1':et_columns1, 'et_columns2':et_columns2, 'et_columns3':et_columns3}


all_columns = {'B-Alert Decontaminated EEG' : {'eeg_current_columns1':eeg_current_columns1, 'eeg_current_columns2':eeg_current_columns2, 'eeg_current_columns3':eeg_current_columns3},
               'Eyetracker HTC VIVE Pro Eye' : {'et_columns1':et_columns1, 'et_columns2':et_columns2, 'et_columns3':et_columns3},
                'Shimmer shim exg 02 5F2F ECG': { 'ecg_columns1':ecg_columns1, 'ecg_columns2':ecg_columns2, 'ecg_columns3':ecg_columns3}
              }

groups = []
for i in all_columns:
    groups = groups + list(all_columns[i].keys())

# return the name of the list of groups
def get_key(val):
    for key, value in all_columns.items():
        if val in value:
            return key
 
    return "key doesn't exist"
 

groupstoprint = []
for i in all_columns:
    #print(all_columns[i].keys())
    #print(i)
    for j in all_columns[i]:
        print(j, end=',')
    for k in all_columns[i]:
        for f in all_columns[i][k]:
            print(f)
    groupstoprint = groupstoprint + list(all_columns[i].keys())
    
    

# print the number of each class in dataset
def countClasses(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

# manage the number of classes for experiments - you should modify this function if you need the create multiple classes
def manageClasses(all_Y):
    
    all_Y = np.where(all_Y <=7, 0, all_Y)
    all_Y = np.where(all_Y >7 , 1, all_Y)
  
    #all_Y = np.where(all_Y <=6, 0, all_Y)
    #all_Y = np.where(all_Y ==7 , 1, all_Y)
    #all_Y = np.where(all_Y ==8 , 1, all_Y)
    #all_Y = np.where(all_Y >8, 2, all_Y)

    #all_Y = np.where(all_Y ==10, 3, all_Y)
    

    countClasses(all_Y)
    
    return all_Y

def prepare3DArray(data, all_data_by_event, workedData, events = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']):       
    #events = ['Eyetracker HTC VIVE Pro Eye']
    rownb = 0
    shapes = []
    #workedData = []
    with_problemData = []
    
    for i in range(len(data)):       
        rownb = rownb + data[i].shape[0]
        data[i] = data[i].dropna()
        #print(i)
        #print(data[i]['class'])
        if data[i]['class'].empty:
            print('DataFrame is empty! ', i)
        else:
            #if data[i]['class'][0] < 10:
                #print(data[i].shape[1])
            if data[i].shape[0]/128 >= 2:
                #print(data[i].shape[0])
                class_data = int(data[i].iloc[1,-1:])
                dataTranspose = data[i].tail(256).iloc[:,:-1].T
                dataTranspose['class'] = class_data
                if(data[i].shape[1] <46):
                    with_problemData.append(dataTranspose.values)
                else:
                    workedData.append(dataTranspose.values)

                for idx, key in enumerate(events):
                    events_needed = [key]
                    # filter data by the columns of current event
                    if key in data[i].columns:
                        singleEventData = data[i][key]
                        dataEventTranspose = singleEventData.tail(256).T
                        # Check if all 2D numpy array contains only 0
                        result = np.all((dataEventTranspose.values == 0))
                        if result:
                            print('2D Array contains only 0 i=',i, ' key = '+key)
                            #print(dataEventTranspose)
                        else:
                        #    print('2D Array has non-zero items too')
                            dataEventTranspose['class'] = class_data
                            if key not in all_data_by_event:
                                all_data_by_event[key] = [] #dataTranspose.values
                                #print("not in")
                            #else:
                                #np.append(all_data_by_event[key],dataTranspose.values)
                                #print("in")
                            #print(dataTranspose.shape)
                            all_data_by_event[key].append(dataEventTranspose.values)
                        #print(key+ " i = "+str(i))
                        #print(np.array(all_data_by_event[key]).shape)


    #print(rownb)
    #print(np.min(shapes))
    #print(np.max(shapes))
    return workedData, all_data_by_event

# check if data of selected events is valid and has values
def isValidData(data, eventsCombination):
    isValid = True
    for oneEvent in eventsCombination:
        combinationOneEventData = data[oneEvent]
        isOneEventZeros = np.all((combinationOneEventData.values == 0))

        if isOneEventZeros:
            return False

    nan_Nb = data.isnull().sum().sum()
    if nan_Nb >0:
        return False
    else:
        # Check if all 2D numpy array contains only 0
        result = np.all((data.values == 0))
        if result:
            return False
        else:
            return True
            dataEventsTranspose['class'] = class_data
            combination_name = '_'.join(eventsCombination)
            if combination_name not in all_data_by_event_combination:
                all_data_by_event_combination[combination_name] = [] #dataTranspose.values
                #print("not in")


        all_data_by_event_combination[combination_name].append(dataEventsTranspose.values)

#cut data based window slicing method - with overlap or without, based on parameters values
#window_size_in_sec - correspond to the size of one window in seconds ex: 2 seconds
#data_to_slice_full_time - correspond to the size of data to cut into set of windows ex: 10 seconds (the last 10s to cut from the given data)
#slice_size - correspond to the overlap between windows ex:0.5s 
#sample_rate - correspond to the number of rows per second ex:128
def getDataByWindowsSlicing(data, window_size_in_sec=2, data_to_slice_full_time=2, slice_size=0, sample_rate=128):
    dataArray = []
    
    window_size_by_frame = window_size_in_sec*sample_rate
    data_to_slice_full_time_by_frame = data_to_slice_full_time*sample_rate
    slice_size_by_frame = slice_size*sample_rate
    

    current_window_begin = window_size_by_frame
    current_window_end = 0
    data_size = data.shape[0]

    #check if data size is less then the data to slice, in order to get windows for existing data only
    if( data_size < data_to_slice_full_time_by_frame):
        data_to_slice_full_time_by_frame = data_size
    
    count = 0
    # loop data until it cut all windows in the full slice
    while (current_window_begin<=data_to_slice_full_time_by_frame):
        if current_window_end==0:
            current_window_data = data.iloc[-current_window_begin:]
        else:
            current_window_data = data.iloc[-current_window_begin:-current_window_end]

        dataArray.append(current_window_data)
        count +=1
        current_window_begin = int(current_window_begin+slice_size_by_frame)
        current_window_end = int(current_window_end+slice_size_by_frame)
    return dataArray

# remove data with empty classes
def filterData(data):
    filteredData = []
    for i in range(len(data)):
        #print(data[i]['class'][0])
        if data[i]['class'].empty:
            print(' DataFrame is empty! ', i)
        else:
            filteredData.append(data[i])
    return filteredData

#return dataset for the given combination
def prepare3DArrayByCombinationAndFeatures(data, all_data_by_event_combination, eventsCombination, current_features_to_use):       
    global window_size_in_sec, data_to_slice_full_time, slice_size, sample_rate

    #loop each SOC data
    for i in range(len(data)):
        #check that data is valid
        if data[i]['class'].empty:
            print(' DataFrame is empty! ', i)
        else:
            #check that data has more than 2 seconds
            if data[i].shape[0]/128 >= 2:
                #read class for the current SOC
                class_data = int(data[i].iloc[1,-1:])

                #Get data only if all events combination exist
                if(all(x in list(data[i].columns.levels[0]) for x in eventsCombination)): 
                    

                        
                    combination_name = '_'.join(current_features_to_use)

                    CombinationEventData = data[i].loc[:,(eventsCombination,current_features_to_use)]
                    #cut data to windows based on selected parameters
                    dataArray = getDataByWindowsSlicing(CombinationEventData, window_size_in_sec=window_size_in_sec, data_to_slice_full_time=data_to_slice_full_time, slice_size=slice_size, sample_rate=sample_rate)
                    
                    #loop each window of data, and add it to the list of data of current combination
                    for dataWindow in dataArray:
                        if isValidData(dataWindow, eventsCombination):
                            dataEventsTranspose = dataWindow.T
                            dataEventsTranspose['class'] = class_data
                            if combination_name not in all_data_by_event_combination:
                                all_data_by_event_combination[combination_name] = [] #dataTranspose.values
                        
                            all_data_by_event_combination[combination_name].append(dataEventsTranspose.values)
                    
    return all_data_by_event_combination

def prepare3DArrayByCombination(data, all_data_by_event_combination, eventsCombination = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']):       
    global window_size_in_sec, data_to_slice_full_time, slice_size, sample_rate
    
    for i in range(len(data)):
        #print('data ',i)
        if data[i]['class'].empty:
            print(' DataFrame is empty! ', i)
        else:
            #if data[i]['class'][0] < 10:
                #print(data[i].shape[1])
            if data[i].shape[0]/128 >= 2:
                #print(data[i].shape[0])
                class_data = int(data[i].iloc[1,-1:])
                #if eventsCombination in data[i]:

                #Get data only if all events combination exist
                if(all(x in list(data[i].columns.levels[0]) for x in eventsCombination)): 
                    

                        
                    combination_name = '_'.join(eventsCombination)
                    #print(combination_name)
                    CombinationEventData = data[i][eventsCombination]
                    dataArray = getDataByWindowsSlicing(CombinationEventData, window_size_in_sec=window_size_in_sec, data_to_slice_full_time=data_to_slice_full_time, slice_size=slice_size, sample_rate=sample_rate)         
                    for dataWindow in dataArray:
                        if isValidData(dataWindow, eventsCombination):
                            dataEventsTranspose = dataWindow.T
                            dataEventsTranspose['class'] = class_data
                            if combination_name not in all_data_by_event_combination:
                                all_data_by_event_combination[combination_name] = [] #dataTranspose.values
                        
                            all_data_by_event_combination[combination_name].append(dataEventsTranspose.values)
                    
    return all_data_by_event_combination

def prepare3DArrayByCombination_initial(data, all_data_by_event_combination, eventsCombination = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']):       
    for i in range(len(data)):       
        if data[i]['class'].empty:
            print('DataFrame is empty! ', i)
        else:
            #if data[i]['class'][0] < 10:
                #print(data[i].shape[1])
            if data[i].shape[0]/128 >= 2:
                #print(data[i].shape[0])
                class_data = int(data[i].iloc[1,-1:])
                #if eventsCombination in data[i]:

                if(all(x in list(data[i].columns.levels[0]) for x in eventsCombination)): 
                    

                        

                #if data[i].columns.isin(eventsCombination, level=0).all():
                    #print(eventsCombination)
                    CombinationEventData = data[i][eventsCombination]
                    
                    
                    #dataEvents = CombinationEventData.tail(256).iloc[:,:-1]
                    dataEvents = CombinationEventData.tail(256)
                    #print(CombinationEventData.shape, ' : ', dataEvents.shape)

                    for oneEvent in eventsCombination:
                        combinationOneEventData = dataEvents[oneEvent]
                        isOneEventZeros = np.all((combinationOneEventData.values == 0))
                        #print('isOneEventZeros ', isOneEventZeros)
                        #print(combinationOneEventData)
                        if isOneEventZeros==True:
                            print("you should skip this post")
                        to_skip = False
                        if isOneEventZeros:
                            #print('2D Array contains of one event only 0 i=',i, ' event = '+oneEvent)
                            to_skip = True
                            #print(dataEventTranspose)
                        #else:
                    #dataEventsTranspose = CombinationEventData.tail(256).iloc[:,:-1].T
                    dataEventsTranspose = dataEvents.T

                    nan_Nb = dataEventsTranspose.isnull().sum().sum()
                    if nan_Nb >0 or to_skip:
                        #print(nan_Nb, '     to skip:',)
                        dataEventsTranspose = dataEventsTranspose.dropna()
                    else:
                        # Check if all 2D numpy array contains only 0
                        #print(dataEventsTranspose.columns)
                        result = np.all((dataEventsTranspose.values == 0))
                        if result:
                            print('2D Array contains only 0 i=',i, ' Events = ',eventsCombination)
                            #print(dataEventTranspose)
                        else:
                        #    print('2D Array has non-zero items too')
                            dataEventsTranspose['class'] = class_data
                            combination_name = '_'.join(eventsCombination)
                            if combination_name not in all_data_by_event_combination:
                                all_data_by_event_combination[combination_name] = [] #dataTranspose.values
                                #print("not in")
                        
                        
                        all_data_by_event_combination[combination_name].append(dataEventsTranspose.values)
    return all_data_by_event_combination

def prepare3DArrayByCombination_old(data, all_data_by_event_combination, eventsCombination = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']):       
    for i in range(len(data)):       
        if data[i]['class'].empty:
            print('DataFrame is empty! ', i)
        else:
            #if data[i]['class'][0] < 10:
                #print(data[i].shape[1])
            if data[i].shape[0]/128 >= 2:
                #print(data[i].shape[0])
                class_data = int(data[i].iloc[1,-1:])
                #if eventsCombination in data[i]:

                if(all(x in list(data[i].columns.levels[0]) for x in eventsCombination)): 

                #if data[i].columns.isin(eventsCombination, level=0).all():
                    #print(eventsCombination)
                    CombinationEventData = data[i][eventsCombination]

                    #dataEventsTranspose = CombinationEventData.tail(256).iloc[:,:-1].T
                    dataEventsTranspose = CombinationEventData.tail(256).T
                    # Check if all 2D numpy array contains only 0
                    result = np.all((dataEventsTranspose.values == 0))
                    if result:
                        print('2D Array contains only 0 i=',i, ' key = '+key)
                        #print(dataEventTranspose)
                    else:
                    #    print('2D Array has non-zero items too')
                        dataEventsTranspose['class'] = class_data
                        combination_name = '_'.join(eventsCombination)
                        if combination_name not in all_data_by_event_combination:
                            all_data_by_event_combination[combination_name] = [] #dataTranspose.values
                            #print("not in")
                        all_data_by_event_combination[combination_name].append(dataEventsTranspose.values)

    return all_data_by_event_combination
def loadDataFromSavedArraysByUser():
    all_data_by_user = {}
    #sequence_list = []
    path = '/home/deep01/zaher/exportedObject/normal/'
    listOfFiles = os.listdir(path)
    pattern = "*_data.array"
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            userId = entry.split('_')[0];
            #print (userId)
            if userId not in all_data_by_user:
                all_data_by_user[userId] = [] 
                
            single_data = pd.read_pickle(path+entry)
            #print(len(single_data))

            all_data_by_user[userId] = np.concatenate((all_data_by_user[userId],single_data))
    return all_data_by_user
            
def loadDataFromSavedArrays():
    data = pd.read_pickle("/home/deep01/zaher/all_data_full_features_10_classes.array")
    data.shape
    data1 = pd.read_pickle("/home/deep01/zaher/all_data_full_features_with_problem_10_classes.array")
    data1.shape
    data2 = pd.read_pickle("/home/deep01/zaher/all_data_full_features_exception_10_classes.array")
    data2.shape
    return data, data1, data2


def prepareAllDataArrays(data, data1, data2):
    all_data_by_event = {}
    workedData = []
    workedData, all_data_by_event = prepare3DArray(data,all_data_by_event, workedData)
    workedData, all_data_by_event = prepare3DArray(data1,all_data_by_event, workedData)
    workedData, all_data_by_event = prepare3DArray(data2, all_data_by_event, workedData)
    
def prepareAllDataArraysByCombination(data, data1, data2):
    all_data_by_event_combination = {}
    events = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']
    count = 0
    for L in range(1,len(events)+1):
        for subset in itertools.combinations(events, L):
            eventsCombination = list(subset)
            #print(eventsCombination, "  :  ", len(eventsCombination))
            all_data_by_event_combination = prepare3DArrayByCombination(data, all_data_by_event_combination, eventsCombination)
            all_data_by_event_combination = prepare3DArrayByCombination(data1, all_data_by_event_combination, eventsCombination)
            all_data_by_event_combination = prepare3DArrayByCombination(data2, all_data_by_event_combination, eventsCombination)
    
    return all_data_by_event_combination

# return a list of dataset for all combination of events
def prepareAllSplitedDataArraysByCombination(data):
    all_data_by_event_combination = {}
    events = ['B-Alert Decontaminated EEG', 'B-Alert EEG',   'Eyetracker HTC VIVE Pro Eye','R Analysis GazeAnalysis I-VT filter','Shimmer GSR 4B59', 'Shimmer shim exg 02 5F2F ECG']
    count = 0
    for L in range(1,len(events)+1):
        for subset in itertools.combinations(events, L):
            eventsCombination = list(subset)
            #print(eventsCombination, "  :  ", len(eventsCombination))
            all_data_by_event_combination = prepare3DArrayByCombination(data, all_data_by_event_combination, eventsCombination)

    return all_data_by_event_combination

# return a list of dataset for all combination of groups
def prepareAllSplitedDataArraysByCombinationAndFeatures(data, groups):
    all_data_by_event_combination_features = {}
    count = 0
    # loop al combination
    for L in range(2,len(groups)+1):
        for subset in itertools.combinations(groups, L):
            # list of one combination
            eventsCombination = list(subset)
            current_columns = []
            current_combination_columns_groups = {}
            #create the list columns for this combination
            for oneColumnsSet in eventsCombination:
                currentEvent = get_key(oneColumnsSet)
                if currentEvent in current_combination_columns_groups:
                    current_combination_columns_groups[currentEvent] = current_combination_columns_groups[currentEvent] + all_columns_groups[oneColumnsSet]
                else:
                    current_combination_columns_groups[currentEvent] = all_columns_groups[oneColumnsSet]

                current_columns =current_columns+  all_columns_groups[oneColumnsSet]
            #return dataset for the current combination
            all_data_by_event_combination_features = prepare3DArrayByCombinationAndFeatures(data, all_data_by_event_combination_features, list(current_combination_columns_groups.keys()), current_columns) 

            print(count)
            print(current_columns)
            print("---------------------------------------------------------------------")
            count +=1


    return all_data_by_event_combination_features

def prepareAllDataArraysByOneCombination(data, data1, data2, eventsCombination):
    all_data_by_event_combination = {}

    all_data_by_event_combination = prepare3DArrayByCombination(data, all_data_by_event_combination, eventsCombination)
    all_data_by_event_combination = prepare3DArrayByCombination(data1, all_data_by_event_combination, eventsCombination)
    all_data_by_event_combination = prepare3DArrayByCombination(data2, all_data_by_event_combination, eventsCombination)

    return all_data_by_event_combination

def exportMLSTMData(X, Y, path):
    global data_folder_name
    initial_path = "/home/deep01/zaher/MLSTM-FCN-master/"+data_folder_name+"/"
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #y_train = y_train
    #y_test = y_test[:,0]
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    # ''' Save the datasets '''
    #print("Train dataset : ", X_train.shape, y_train.shape)
    #print("Test dataset : ", X_test.shape, y_test.shape)
    #print("Train dataset metrics : ", X_train.mean(), X_train.std())
    #print("Test dataset : ", X_test.mean(), X_test.std())
    #print("Nb classes : ", len(np.unique(y_train)))
    
    countClasses(y_train)
    
    countClasses(y_test)
    
    path = initial_path+path

    if not os.path.exists(path):
        os.makedirs(path)
    
    #print(path)
    np.save(path + '/X_train.npy', X_train)
    np.save(path + '/y_train.npy', y_train)
    np.save(path + '/X_test.npy', X_test)
    np.save(path + '/y_test.npy', y_test)
    
def exportMLSTMDataWithoutSplit(X, Y, path, Datatype= "train"):
    global data_folder_name
    initial_path = "/home/deep01/zaher/MLSTM-FCN-master/"+data_folder_name+"/"
    
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #y_train = y_train
    #y_test = y_test[:,0]
    Y = Y.reshape(Y.shape[0],1)
    #y_test = y_test.reshape(y_test.shape[0],1)
    # ''' Save the datasets '''
    #print(Datatype+" Train dataset : ", X.shape, Y.shape)
    #print(Datatype+" dataset metrics : ", X.mean(), X.std())
    #print("Nb classes : ", len(np.unique(Y)))
    
    countClasses(Y)
    
    #countClasses(y_test)
    
    path = initial_path+path

    if not os.path.exists(path):
        os.makedirs(path)
    
    #print(path)
    np.save(path + '/X_'+Datatype+'.npy', X)
    np.save(path + '/y_'+Datatype+'.npy', Y)
    #np.save(path + '/X_test.npy', X_test)
    #np.save(path + '/y_test.npy', y_test)

def exportDataByevent(all_data_by_event):
    for idx, key in enumerate(all_data_by_event.keys()):
        #print(key)
        #print(np.array(all_data_by_event[key]).shape)
        
        workedDataNP = np.array(all_data_by_event[key])
        X = workedDataNP[:,:,:-1]
        Y = workedDataNP[:,0,-1]
        #print(X.shape)
        #print(Y.shape)
        #print(Y)
        Y = manageClasses(Y)
        exportMLSTMData(X, Y, path=key)

def exportDataByeventWithoutSplit(all_data_by_event, Datatype="train"):
    for idx, key in enumerate(all_data_by_event.keys()):
        #print(key)
        #print(np.array(all_data_by_event[key]).shape)
        
        workedDataNP = np.array(all_data_by_event[key])
        X = workedDataNP[:,:,:-1]
        Y = workedDataNP[:,0,-1]
        
        #indexArr = np.argwhere(Y == 10)
        #print(X.shape)
        #print(Y.shape)
        
        #Y = np.delete(Y, indexArr)
        #X = np.delete(X, indexArr, axis=0)

        #print(X.shape)
        #print(Y.shape)

        #print(Y)
        Y = manageClasses(Y)
        #path = key
        path = "model"+str(idx)
        exportMLSTMDataWithoutSplit(X, Y, path=path, Datatype=Datatype)
    
def exportDataAllCompleteData(workedData):
    workedDataNP = np.array(workedData)
    #print(workedDataNP.shape)
    X = workedDataNP[:,:,:-1]
    Y = workedDataNP[:,0,-1]
    
    #print(X.shape)
    #print(Y.shape)
    #print(Y)
    Y = manageClasses(Y)
    exportMLSTMData(X, Y, path="zaher")

def splitData(data, Train_rate = 0.8):
    AllfilterData = filterData(data)
    trainingSize = int(len(AllfilterData)*0.8)
    #print(trainingSize)
    np.random.shuffle(AllfilterData)
    return AllfilterData[:trainingSize], AllfilterData[trainingSize:]


# create a dataset (train and test) ready for deep learning model - with all combination of events
# You can creade many folds for the same dataset randomly splitted using "nb_fold" parameter
# 
def crossValSplitForEventCombination(nb_fold, data, folder_name_prefix="_data_no_augmentation_2C_byPost_all_new"):
    global data_folder_name
    export_path = '/home/deep01/zaher/MLSTM-FCN-master/output/'
    for i in range(nb_fold):
        # split this fold randomly to train and test
        training, test = splitData(data, Train_rate = 0.8)
        
        # return a list of train dataset for all events combination
        all_training_by_event_combination_n = prepareAllSplitedDataArraysByCombination(training)
        # return a list of test dataset for all events combination
        all_test_by_event_combination_n = prepareAllSplitedDataArraysByCombination(test)
        
        # export each combination in a repository for this fold(train and test together)
        data_folder_name = "output/fold"+str(i)+folder_name_prefix
        exportDataByeventWithoutSplit(all_training_by_event_combination_n, "train")
        exportDataByeventWithoutSplit(all_test_by_event_combination_n, "test")
        
        # save both lists in a pickle
        pickleName =  'all_training_by_event_combination_overlap_fold'+str(i)+'.array'
        pickleFullPath = export_path + pickleName
        with open(pickleFullPath, 'wb') as all_training_by_event_combination_file:
            pickle.dump(all_training_by_event_combination_n, all_training_by_event_combination_file)
            
        pickleName =  'all_test_by_event_combination_overlap_fold'+str(i)+'.array'
        pickleFullPath = export_path + pickleName
        with open(pickleFullPath, 'wb') as all_test_by_event_combination_file:
            pickle.dump(all_test_by_event_combination_n, all_test_by_event_combination_file)
            

# create a dataset (train and test) ready for deep learning model - with all combination of groups of features
# You can creade many folds for the same dataset randomly splitted using "nb_fold" parameter
# 
def crossValSplitForFeaturesCombination(nb_fold, data, groups, folder_name_prefix="_data_no_augmentation_2C_byPost_all_new"):
    global data_folder_name
    export_path = '/home/deep01/zaher/MLSTM-FCN-master/output/'
    for i in range(nb_fold):
        print("fold "+str(i))
        # split this fold randomly to train and test
        training, test = splitData(data, Train_rate = 0.8)
        
        # return a list of train dataset for all combination
        all_training_by_event_combination_n = prepareAllSplitedDataArraysByCombinationAndFeatures(training, groups)
        # return a list of test dataset for all combination
        all_test_by_event_combination_n = prepareAllSplitedDataArraysByCombinationAndFeatures(test, groups)
        
        # export each combination in a repository for this fold(train and test together)
        data_folder_name = "output/fold"+str(i)+folder_name_prefix
        exportDataByeventWithoutSplit(all_training_by_event_combination_n, "train")
        exportDataByeventWithoutSplit(all_test_by_event_combination_n, "test")
        
        # save both lists in a pickle
        pickleName =  'all_training_by_event_combination_by_features_overlap_'+str(folder_name_prefix)+'_fold'+str(i)+'.array'
        pickleFullPath = export_path + pickleName
        with open(pickleFullPath, 'wb') as all_training_by_event_combination_file:
            pickle.dump(all_training_by_event_combination_n, all_training_by_event_combination_file)
            
        pickleName =  'all_test_by_event_combination_by_features_overlap_'+str(folder_name_prefix)+'fold'+str(i)+'.array'
        pickleFullPath = export_path + pickleName
        with open(pickleFullPath, 'wb') as all_test_by_event_combination_file:
            pickle.dump(all_test_by_event_combination_n, all_test_by_event_combination_file)








#data = pd.read_pickle("/home/deep01/zaher/all_data_full_features_10_classes.array")
#data1 = pd.read_pickle("/home/deep01/zaher/all_data_full_features_manuel_annotation_10_classes.array")
#AllData = np.concatenate((data,data1))

AllData = pd.read_pickle("/home/deep01/zaher/all_data_full_features_10_classes_final.array")


# without Overlap train and test
window_size_in_sec=2
data_to_slice_full_time=2
slice_size=0.5
sample_rate=128
data_folder_name = ""
#crossValSplitForFeaturesCombination(1, AllData, groups, folder_name_prefix= "_data_no_augmentation_2C_byFeatures_all_new")



# with Overlap train and test
window_size_in_sec=2
data_to_slice_full_time=10
slice_size=0.5
sample_rate=128
data_folder_name = ""
folds_number = 2
crossValSplitForFeaturesCombination(folds_number, AllData, groups, folder_name_prefix= "_data_with_augmentation_2C_byFeatures_all_new")