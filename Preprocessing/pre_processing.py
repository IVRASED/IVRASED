import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import seaborn as sns
import matplotlib
import matplotlib.animation as animation

from hhpy.plotting import animplot

import datetime
from pandas import datetime as dt
from matplotlib import pyplot

import os, fnmatch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle


#List of events (sensors) and features (columns) in csv file
###

TIME_STAMP_COLUMN = 'iMotions_Synchronization_Timestamp'
STUDY_INFO = {
    # 'StudyName': str,
    # 'Name': str,
    # 'Age': int,
    # 'Gender': str,
    # 'EventSource': str,
    # 'Row': np.float64,
    'iMotions_Synchronization_Timestamp': np.float64,
    # 'Annotations_Screenrecording-1_Consultsheet_Before_Active': str,
    # 'Annotations_Screenrecording-1_Consultsheet_Before_Instance': np.float64,
    # 'Annotations_Screenrecording-1_Poste_Active': str,
    # 'Annotations_Screenrecording-1_Poste_Comment': str,
    # 'Annotations_Screenrecording-1_Poste_Instance': np.float64,
    # 'Eventsource_External_Events_API_(V0)': bool,
    # 'Api_Markername': str,
    # 'Api_Markerdescription': str,
    # 'Api_Markertype': str,
    # 'Api_Scenetype': str

}
# Normal annotation - when every think works well
Annotation = {
    'Eventsource_External_Events_API_(V0)': bool,
    'Api_Markername': str,
    'Api_Markerdescription': str,
    'Api_Markertype': str,
    # 'Api_Scenetype': str
}
# Manual annotation when API is disconnected
Manual_Annotation = {
    'Annotations_Screenrecording-1_Consultsheet_Before_Active': str,
    'Annotations_Screenrecording-1_Consultsheet_Before_Instance': np.float64,
    'Annotations_Screenrecording-1_Poste_Active': str,
    'Annotations_Screenrecording-1_Poste_Comment': str,
    'Annotations_Screenrecording-1_Poste_Instance': np.float64
}
#Manuel annotation for the first 2 participants
Exception_Manual_Annotation = {
    'Annotations_Screenrecording-1_Competence_Active': str,
    'Annotations_Screenrecording-1_Competence_Comment': np.float64,
    'Annotations_Screenrecording-1_Competence_Instance': np.float64
}

B_Alert_BrainState = {
    'Eventsource_B-Alert_Brainstate': bool,
    # 'EEG_Metric_Epoch': np.float64,
    # 'EEG_Metric_Hour': np.float64,
    # 'EEG_Metric_Min': np.float64,
    # 'EEG_Metric_Sec': np.float64,
    # 'EEG_Metric_Milli': np.float64,
    'EEG_Metric_Classification': np.float64,
    'EEG_Metric_High_Engagement': np.float64,
    'EEG_Metric_Low_Engagement': np.float64,
    'EEG_Metric_Distraction': np.float64,
    'EEG_Metric_Drowsy': np.float64,
    'EEG_Metric_Workload_FBDS': np.float64,
    'EEG_Metric_Workload_BDS': np.float64,
    'EEG_Metric_Workload_Average': np.float64
}

B_Alert_Decontaminated_EEG = {
    'Eventsource_B-Alert_Decontaminated_EEG': bool,
    # 'EEG_Decon_Epoch': np.float64,
    # 'EEG_Decon_Offset': np.float64,
    # 'EEG_Decon_Hour': np.float64,
    # 'EEG_Decon_Min': np.float64,
    # 'EEG_Decon_Sec': np.float64,
    # 'EEG_Decon_Milli': np.float64,
    # 'EEG_Decon_ECG': np.float64,
    'EEG_Decon_Poz': np.float64,
    'EEG_Decon_Fz': np.float64,
    'EEG_Decon_Cz': np.float64,
    'EEG_Decon_C3': np.float64,
    'EEG_Decon_C4': np.float64,
    'EEG_Decon_F3': np.float64,
    'EEG_Decon_F4': np.float64,
    'EEG_Decon_P3': np.float64,
    'EEG_Decon_P4': np.float64

}

B_Alert_EEG = {
    'Eventsource_B-Alert_EEG': bool,
    # 'EEG_Epoch': np.float64,
    # 'EEG_Offset': np.float64,
    # 'EEG_Hour': np.float64,
    # 'EEG_Min': np.float64,
    # 'EEG_Sec': np.float64,
    # 'EEG_Milli': np.float64,
    # 'EEG_ECG': np.float64,
    'EEG_Poz': np.float64,
    'EEG_Fz': np.float64,
    'EEG_Cz': np.float64,
    'EEG_C3': np.float64,
    'EEG_C4': np.float64,
    'EEG_F3': np.float64,
    'EEG_F4': np.float64,
    'EEG_P3': np.float64,
    'EEG_P4': np.float64
}
Eyetracker_HTC_VIVE_Pro_Eye = {
    'Eventsource_Eyetracker_HTC_VIVE_Pro_Eye': bool,
    'ET_Gaze_2D_ET_Gazeleftx': np.float64,
    'ET_Gaze_2D_ET_Gazelefty': np.float64,
    'ET_Gaze_2D_ET_Gazerightx': np.float64,
    'ET_Gaze_2D_ET_Gazerighty': np.float64,
    'ET_Pupil_ET_Pupilleft': np.float64,
    'ET_Pupil_ET_Pupilright': np.float64,
    'ET_Other_ET_Timesignal': np.float64,
    'ET_Distance_ET_Distanceleft': np.float64,
    'ET_Distance_ET_Distanceright': np.float64,
    'ET_Capture_ET_Cameraleftx': np.float64,
    'ET_Capture_ET_Cameralefty': np.float64,
    'ET_Capture_ET_Camerarightx': np.float64,
    'ET_Capture_ET_Camerarighty': np.float64,
    'ET_Head_ET_Headrotationx': np.float64,
    'ET_Head_ET_Headrotationy': np.float64,
    'ET_Head_ET_Headrotationz': np.float64,
    'ET_Head_ET_Headpositionvectorx': np.float64,
    'ET_Head_ET_Headpositionvectory': np.float64,
    'ET_Head_ET_Headpositionvectorz': np.float64,
    'ET_Head_ET_Headvelocityx': np.float64,
    'ET_Head_ET_Headvelocityy': np.float64,
    'ET_Head_ET_Headvelocityz': np.float64,
    'ET_Head_ET_Headangularvelocityx': np.float64,
    'ET_Head_ET_Headangularvelocityy': np.float64,
    'ET_Head_ET_Headangularvelocityz': np.float64,
    'ET_Expression_ET_Lefteyeopenness': np.float64,
    'ET_Expression_ET_Righteyeopenness': np.float64,
    'ET_Expression_ET_Lefteyesqueeze': np.float64,
    'ET_Expression_ET_Righteyesqueeze': np.float64,
    'ET_Expression_ET_Lefteyefrown': np.float64,
    'ET_Expression_ET_Righteyefrown': np.float64,
    # 'ET_Other_ET_VR_Headsetconnectedstate': np.float64,
    # 'ET_Other_ET_VR_Userpresentstate': np.float64
}
R_Analysis_Gazeanalysis_I_VT_Filter = {
    'Eventsource_R_Analysis_Gazeanalysis_I-VT_Filter': bool,

    'Gaze X': np.float64,
    'Gaze Y': np.float64,
    'Interpolated Gaze X': np.float64,
    'Interpolated Gaze Y': np.float64,
    'Interpolated Distance': np.float64,
    'Gaze Velocity': np.float64,
    'Gaze Acceleration': np.float64,
    'Fixation Index': np.float64,
    'Fixation Index by Stimulus': np.float64,
    'Fixation X': np.float64,
    'Fixation Y': np.float64,
    'Fixation Start': np.float64,
    'Fixation End': np.float64,
    'Fixation Duration': np.float64,
    'Fixation Dispersion': np.float64,
    'Saccade Index': np.float64,
    'Saccade Index by Stimulus': np.float64,
    'Saccade Start': np.float64,
    'Saccade End': np.float64,
    'Saccade Duration': np.float64,
    'Saccade Amplitude': np.float64,
    'Saccade Peak Velocity': np.float64,
    'Saccade Peak Acceleration': np.float64,
    'Saccade Peak Deceleration': np.float64,
    'Saccade Direction': np.float64

}
Eventsource_Shimmer_GSR_4B59 = {
    'Eventsource_Shimmer_GSR_4B59': bool,
    #'Other_Timestamp_RAW_Shimmer_GSR': np.float64,
    #'Other_Timestamp_CAL_Shimmer_GSR': np.float64,
    #'Other_System_Timestamp_CAL_Shimmer_GSR': np.float64,
    'Other_Vsensebatt_RAW_Shimmer_GSR': np.float64,
    'Other_Vsensebatt_CAL_Shimmer_GSR': np.float64,
    'PPG_Bloodvolumepulse_RAW_Shimmer_GSR': np.float64,
    'PPG_Bloodvolumepulse_CAL_Shimmer_GSR': np.float64,
    'Other_GSR_RAW_Shimmer_GSR': np.float64,
    'GSR_Resistance_CAL_Shimmer_GSR': np.float64,
    'GSR_Conductance_CAL_Shimmer_GSR': np.float64,
    'PPG_Metric_Interbeatinterval_Shimmer_GSR': np.float64,
    #'Other_Packet_Reception_Rate_RAW_Shimmer_GSR': np.float64 # > 40
}

Shimmer_Shim_Exg_02_5F2F_ECG = {
    'Eventsource_Shimmer_Shim_Exg_02_5F2F_ECG': bool,

    # 'Other_Timestamp_RAW_Shim_Exg_02': np.float64,
    # 'Other_Timestamp_CAL_Shim_Exg_02': np.float64,
    # 'Other_System_Timestamp_CAL_Shim_Exg_02': np.float64,
     'Other_Vsensebatt_RAW_Shim_Exg_02': np.float64,
     'Other_Vsensebatt_CAL_Shim_Exg_02': np.float64,
     'Other_EXG1_Status_RAW_Shim_Exg_02': np.float64,

    'Other_ECG_LL-RA_RAW_Shim_Exg_02': np.float64,
    'Other_ECG_LL-RA_CAL_Shim_Exg_02': np.float64,
    'Other_ECG_LA-RA_RAW_Shim_Exg_02': np.float64,
    'Other_ECG_LA-RA_CAL_Shim_Exg_02': np.float64,
    'Other_EXG2_Status_RAW_Shim_Exg_02': np.float64,
    'Other_ECG_Vx-RL_RAW_Shim_Exg_02': np.float64,
    'Other_ECG_Vx-RL_CAL_Shim_Exg_02': np.float64,
    'Other_Heart_Rate_ECG_LL-RA_ALG_Shim_Exg_02': np.float64,
    'Other_IBI_ECG_LL-RA_ALG_Shim_Exg_02': np.float64,
    # 'Other_Packet_Reception_Rate_RAW_Shim_Exg_02': np.float64

}

EventNames = {
    # 'B-Alert BrainState' : B_Alert_BrainState,
    'B-Alert Decontaminated EEG': B_Alert_Decontaminated_EEG,
    'B-Alert EEG': B_Alert_EEG,
    'Eyetracker HTC VIVE Pro Eye': Eyetracker_HTC_VIVE_Pro_Eye,
    'R Analysis GazeAnalysis I-VT filter': R_Analysis_Gazeanalysis_I_VT_Filter,
    'Shimmer GSR 4B59': Eventsource_Shimmer_GSR_4B59,
    'Shimmer shim exg 02 5F2F ECG': Shimmer_Shim_Exg_02_5F2F_ECG

}
EventFrequencies = {
    'B-Alert BrainState': 1,
    'B-Alert Decontaminated EEG': 256,
    'B-Alert EEG': 256,
    'Eyetracker HTC VIVE Pro Eye': 120,
    'R Analysis GazeAnalysis I-VT filter': 120,
    'Shimmer GSR 4B59': 128,
    'Shimmer shim exg 02 5F2F ECG': 512

}

# return the types of list of events
def generate_types(events_needed):
    types = {}
    types.clear()

    for e in events_needed:
        types.update(EventNames[e])
    types.update(STUDY_INFO)

    return types

# return the types of all events
def generate_all_types():
    all_types = {}

    for e in EventNames:
        all_types.update(EventNames[e])
    all_types.update(STUDY_INFO)

    return all_types

# used to parse timestamp from csv file
def parser(x):
    try:
        return dt.strptime(str(datetime.timedelta(seconds=np.multiply(float(x), 0.001))), '%H:%M:%S.%f')
    except ValueError:
        return dt.strptime(str(datetime.timedelta(seconds=np.multiply(float(x), 0.001))), '%H:%M:%S')

# return respondant data from a selected csv file
def read_respondant_data(filename):
    data_respondent = pd.read_csv(filename, sep=',', index_col=0, encoding='utf-8', nrows=5, usecols=[0, 1],
                                  header=None, skiprows=[0])
    data_respondent = data_respondent.to_dict()[1]
    data_respondent['name'] = data_respondent['#Respondent Name'][0:5]
    data_respondent['sequence_number'] = data_respondent['#Respondent Name'][7:]
    return data_respondent

# return the two features' headers line number from a selected csv file
def get_header_lines(filename):
    file_headers = pd.read_csv(filename, sep=',', encoding='utf-8', nrows=50, usecols=[0], header=None)
    first_header_row = file_headers[file_headers[0] == '#Channel identifier'].index[0]
    second_header_row = file_headers[file_headers[0] == 'Row'].index[0]

    return first_header_row, second_header_row

# read all data from a csv file
def read_data_from_file(filename, first_header_line=31, second_header_line=34):
    data = pd.read_csv(filename, sep=',', header=[first_header_line, second_header_line], encoding='utf-8',
                       skip_blank_lines=True)
    #if features' header does not exist in first header line, use the second header line instead
    data.columns = [x[1] if "Unnamed" in x[0] else x[0] for x in data.columns]
    # parse timestamp
    data['dateTimeIndex'] = [parser(x) for x in data[TIME_STAMP_COLUMN]]
    # transform data to series
    data.squeeze()
    # set parsed timestamp as index for data
    data = data.set_index('dateTimeIndex')

    return data

# return a list of eixsting events and a list of missing events in a csv file
def get_existing_missing_events(filename):
    file_events = pd.read_csv(filename, sep=',', encoding='utf-8', nrows=8, usecols=[1], header=None, skiprows=10)

    existing_events = []
    missing_events = []

    events = list(EventNames.keys())
    for idx, key in enumerate(events):

        found = file_events[file_events[1].str.contains(key, na=False)]

        if found.count()[1]:
            # print('exist')
            existing_events.append(key)
        else:
            # print('not exist')
            missing_events.append(key)

    return existing_events, missing_events

# filter data by a specific list of events
def getDataByEvent(data_tmp, events, TimeByFileInMinutes=0):
    data = {}
    # events = list(EventNames.keys())
    for idx, key in enumerate(events):
        print(idx)
        events_needed = [key]
        print(events_needed)
        # get only the columns of current event
        types = generate_types(events_needed)
        columns = list()
        columns.clear()
        columns = list(types.keys())

        # filter data by the columns of current event
        singleEventData = data_tmp[columns]

        # filter data to keep only the lines with event_source=1 of current event
        singleEventData = singleEventData[singleEventData[columns[0]] == True]

        if TimeByFileInMinutes > 0:
            # some time / sample rate stats
            TimeByEventInMinutes = singleEventData.shape[0] / (EventFrequencies[key] * 60)
            # get the real sample rate of current event
            realSampleRate = singleEventData.shape[0] / (TimeByFileInMinutes * 60)
            TimeByEventInMinutesRealSampleRate = singleEventData.shape[0] / (realSampleRate * 60)
            print("TimeByEventInMinutes: %s realSampleRate:%s TimeByEventInMinutesRealSampleRate:%s" % (
                str(TimeByEventInMinutes), str(realSampleRate), str(TimeByEventInMinutesRealSampleRate)))

        # resample data to 128 hz
        print(singleEventData.shape)
        if realSampleRate >= 128:
            singleEventData = pd.DataFrame(singleEventData.resample('7.8125ms').mean())
        else:
            singleEventData = pd.DataFrame(singleEventData.resample('7.8125ms').pad().interpolate('linear'))
        print(singleEventData.shape)

        # remove unneeded columns
        singleEventData.drop([singleEventData.columns[0], singleEventData.columns[-1]], axis=1, inplace=True)

        # uncomment the next line if you want to export the data of each event in a file
        # singleEventData.to_csv('data_by_events/'+str(key)+'.csv')

        # data["data"+str(idx)] = singleEventData
        data[key] = singleEventData
    return data


# filter only data contained annotation
def getAnnotationData(data_tmp, events):
    annotation_data_list = []
    for idx, event_needed in enumerate(events):
        columns = list()
        columns.clear()
        columns = list(event_needed.keys())
        columns.append(TIME_STAMP_COLUMN)

        simpleEventData = data_tmp[columns]
        simpleEventData = simpleEventData.dropna(axis=0)
        annotation_data_list.append(simpleEventData)

    annotation_data = pd.concat(annotation_data_list, sort=True)
    annotation_data = annotation_data.sort_index()

    return annotation_data

# return the start and the end time of consulting a sheet
def getBeginEndConsultsheet(annotation_data, is_both_annotation_events, begin_first_evaluation, last_finish_evaluation,
                        annotation_dict, i=0):
    # if there is a value before and after assembly 
    if is_both_annotation_events:
        # filter data based on the time of consulting sheet 
        newres = annotation_data.loc[(annotation_data[
                                          'Annotations_Screenrecording-1_Consultsheet_Before_Active'] == 'consultsheet_before') & (
                                             annotation_data[TIME_STAMP_COLUMN] < begin_first_evaluation) & (
                                             annotation_data[TIME_STAMP_COLUMN] > last_finish_evaluation)]
        is_empty = newres.empty

        if is_empty:
            return annotation_dict, is_empty
        begin_consultsheet = newres[TIME_STAMP_COLUMN].min()
        finish_consultsheet = newres[TIME_STAMP_COLUMN].max()

        begin_consultsheet_index = newres.index.min()
        finish_consultsheet_index = newres.index.max()
        # read current post name
        current_poste_name = newres['Annotations_Screenrecording-1_Poste_Comment'][0]

        annotation_dict[current_poste_name] = {}
        annotation_dict[current_poste_name]['begin_consultsheet'] = begin_consultsheet
        annotation_dict[current_poste_name]['finish_consultsheet'] = finish_consultsheet
    # if there is only one value - used in the case of evaluation based on consulting sheet only
    else:
        current_poste_name = 'fin' + str(i)
        annotation_dict[current_poste_name] = {}

    return annotation_dict, current_poste_name, is_empty


# return a dictionary that contains the values and the times of both SOC for all posts
def extractAnnotationData(annotation_data, is_both_annotation_events=True, is_an_exception_manual_annotation=False):
    max_timestamp = annotation_data[TIME_STAMP_COLUMN].max()
    min_timestamp = annotation_data[TIME_STAMP_COLUMN].min()

    last_finish_evaluation = 0
    is_empty = False
    annotation_dict = {}
    i = 0
    #loop on all posts
    while not is_empty:
        i = i + 1
        # if annotation is generated correctly from IMotion 
        if not is_an_exception_manual_annotation:
            #filter data when SOC is displayed before assembly
            res = annotation_data.loc[(annotation_data['Api_Markername'] == 'SOC_DISPLAY') & (
                    annotation_data[TIME_STAMP_COLUMN] > last_finish_evaluation)]
            is_empty = res.empty

            if is_empty:
                continue
            begin_first_evaluation = res[TIME_STAMP_COLUMN].min()
            begin_first_evaluation_index = res.index.min()

            #filter data when SOC is answered before assembly (after consulting sheet)
            res2 = annotation_data.loc[(annotation_data['Api_Markername'] == 'SOC_ANSWER') & (
                    annotation_data[TIME_STAMP_COLUMN] > begin_first_evaluation)]
            is_empty = res2.empty

            if is_empty:
                continue
            finish_first_evaluation = res2[TIME_STAMP_COLUMN].min()

            # read SOC answer before assembly (after consulting sheet)
            first_competence_value = res2['Api_Markerdescription'][0]


            #filter data when SOC is displayed after assembly
            res = annotation_data.loc[(annotation_data['Api_Markername'] == 'SOC_DISPLAY') & (
                    annotation_data[TIME_STAMP_COLUMN] > finish_first_evaluation)]
            is_empty = res.empty

            if is_empty:
                continue
            begin_second_evaluation = res[TIME_STAMP_COLUMN].min()
            begin_second_evaluation_index = res.index.min()

            #filter data when SOC is answered after assembly
            res2 = annotation_data.loc[(annotation_data['Api_Markername'] == 'SOC_ANSWER') & (
                    annotation_data[TIME_STAMP_COLUMN] > begin_second_evaluation)]
            is_empty = res2.empty

            if is_empty:
                continue
            finish_second_evaluation = res2[TIME_STAMP_COLUMN].min()
            finish_second_evaluation_index = res2.index.min()
            
            # read SOC answer after assembly
            second_competence_value = res2['Api_Markerdescription'][0]

            is_empty = res.empty
            if first_competence_value == 'na' or second_competence_value == 'na':
                continue
            annotation_dict, current_poste_name, is_empty = getBeginEndConsultsheet(annotation_data,
                                                                                is_both_annotation_events,
                                                                                begin_first_evaluation,
                                                                                last_finish_evaluation,
                                                                                annotation_dict, i)

            annotation_dict[current_poste_name]['first_competence_value'] = int(first_competence_value)
            annotation_dict[current_poste_name]['second_competence_value'] = int(second_competence_value)
            annotation_dict[current_poste_name]['begin_first_evaluation'] = begin_first_evaluation
            annotation_dict[current_poste_name]['finish_first_evaluation'] = finish_first_evaluation
            annotation_dict[current_poste_name]['begin_second_evaluation'] = begin_second_evaluation
            annotation_dict[current_poste_name]['finish_second_evaluation'] = finish_second_evaluation

            if is_empty:
                continue
            last_finish_evaluation = finish_second_evaluation
        # if annotation is done manualy in IMotion 
        else:
            #loop on all SOC answers
            for posteId in annotation_data['Annotations_Screenrecording-1_Competence_Instance'].unique()[1:]:
                # filter data for the current SOC            
                res = annotation_data.loc[
                    (annotation_data['Annotations_Screenrecording-1_Competence_Instance'] == posteId)]
                # if it is the SOC of consulting sheet (before assembly)
                if (posteId % 2) != 0:
                    begin_first_evaluation = res[TIME_STAMP_COLUMN].min()
                    finish_first_evaluation = res[TIME_STAMP_COLUMN].max()
                    first_competence_value = res['Annotations_Screenrecording-1_Competence_Comment'][0]
                
                # if it is the SOC after the assembly
                else:
                    begin_second_evaluation = res[TIME_STAMP_COLUMN].min()
                    finish_second_evaluation = res[TIME_STAMP_COLUMN].max()
                    second_competence_value = res['Annotations_Screenrecording-1_Competence_Comment'][0]
                    if first_competence_value == 'na' or second_competence_value == 'na':
                        continue
                    annotation_dict, current_poste_name, is_empty = getBeginEndConsultsheet(annotation_data,
                                                                                        is_both_annotation_events,
                                                                                        begin_first_evaluation,
                                                                                        last_finish_evaluation,
                                                                                        annotation_dict)
                    annotation_dict[current_poste_name]['first_competence_value'] = int(first_competence_value)
                    annotation_dict[current_poste_name]['second_competence_value'] = int(second_competence_value)
                    annotation_dict[current_poste_name]['begin_first_evaluation'] = begin_first_evaluation
                    annotation_dict[current_poste_name]['finish_first_evaluation'] = finish_first_evaluation
                    annotation_dict[current_poste_name]['begin_second_evaluation'] = begin_second_evaluation
                    annotation_dict[current_poste_name]['finish_second_evaluation'] = finish_second_evaluation

                    last_finish_evaluation = finish_second_evaluation
            is_empty = True

    return annotation_dict

# filter data for a selected period
def cutDataByStartEnd(data, start_time, end_time, timestamp_field='Timestamp'):
    return data.loc[(data.index >= parser(start_time)) & (data.index <= parser(end_time))]


# used in case we need to reduce the number of classes (originaly 10), it return a value of competence based on the number of classes
def getCompetencesValue(competence, is_binary=False, range=3):
    if is_binary:
        if competence < 6:
            return 0
        else:
            return 1
    else:
        class_numbers = int(10 / range)
        i = 1
        class_limit = i * class_numbers
        while i <= range:
            if competence <= class_limit:
                return i
            i += 1
            class_limit = i * class_numbers

        return (i - 1)

# return a list that contains only the full data of each answer with the class on each row (the answer of the user)
def listDatabyEachAnnotationValue(annotation_dict, all_sampled_data, classes_numbers=10,
                                  is_both_annotation_events=True):
    consultsheet_data_with_class = []
    post_work_data_with_class = []
    all_data_with_class = []
    print(annotation_dict)
    # loop on each post
    for poste_name, details in annotation_dict.items():
        print(poste_name)

        print('---------------------------------------------')
        # if this post has two SOC answers (before and after assembly)
        if is_both_annotation_events:
            begin_consultsheet = details['begin_consultsheet']
            finish_consultsheet = details['finish_consultsheet']
            first_competence_value = details['first_competence_value']

            finish_first_evaluation = details['finish_first_evaluation']
            begin_second_evaluation = details['begin_second_evaluation']
            second_competence_value = details['second_competence_value']
        # if this post has only the SOC answer of consulting sheet
        else:
            begin_consultsheet = details['begin_first_evaluation']
            finish_consultsheet = details['finish_first_evaluation']
            first_competence_value = details['first_competence_value']

            finish_first_evaluation = details['begin_second_evaluation']
            begin_second_evaluation = details['finish_second_evaluation']
            second_competence_value = details['second_competence_value']

        print(parser(begin_consultsheet))
        print(parser(finish_consultsheet))
        print(first_competence_value)
        print('---------------------------------------------')

        # get data of consultsheet 
        consult_sheet_data = cutDataByStartEnd(all_sampled_data, begin_consultsheet, finish_consultsheet,
                                               timestamp_field=TIME_STAMP_COLUMN)
        # add the answer value as class to the data                                       
        consult_sheet_data['class'] = getCompetencesValue(first_competence_value, range=classes_numbers)
        consultsheet_data_with_class.append(consult_sheet_data)
        
        all_data_with_class.append(consult_sheet_data)
        
        # Vincent: ici pour distiguer les données de réponse au SEF après le Consult Sheet de
        # des données après avoir effectué le montage.
        # get data of assembly task 
        post_work_data = cutDataByStartEnd(all_sampled_data, finish_first_evaluation, begin_second_evaluation,
                                           timestamp_field=TIME_STAMP_COLUMN)
        # add the answer value as class to the data                                       
        post_work_data['class'] = getCompetencesValue(second_competence_value, range=classes_numbers)
        post_work_data_with_class.append(post_work_data)
        
        all_data_with_class.append(post_work_data)
        
    return all_data_with_class

# split data to train and test
def split_train_test_data(data_array_with_class):
    all_data_with_class = pd.concat(data_array_with_class)
    list_of_all_data = all_data_with_class.values
    # get class columns
    y = list_of_all_data[:, -1]
    # get all columns except class
    X = list_of_all_data[:, :-1]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# normalize X data
def normalize(X_train, X_test):
    normalizer = preprocessing.MinMaxScaler()
    normalized_train_X = normalizer.fit_transform(X_train)
    normalized_test_X = normalizer.transform(X_test)
    return normalized_train_X, normalized_test_X


# this function should be called to read an IMotion CSV file with annotation
# It returns the data of each SOC answer with its class (answer value)
# "normalise_for_one_file"  : this parameter is used if you can to normalise data based on the value of the selected file
# "split_by_file" : this parameter is used if you need to split the data of the current file to train and test sets
# "is_an_exception_manual_annotation" : this parameter is used if the annotation is done manuely in IMotion 
def runOneFile(filename, export_path = '/home/deep01/zaher/new_export/exportedObjectFullFeatures/', normalise_for_one_file=True, split_by_file=True, is_an_exception_manual_annotation=False):
    data_respondent = {}
    # get user and sequence information
    data_respondent = read_respondant_data(filename)
    current_user_name = data_respondent['name']
    current_sequence_number = data_respondent['sequence_number']

    # get header line number, this is important if not all sensors (events) exist in the file
    first_header_line, second_header_line = get_header_lines(filename)
    # get data from one file
    current_user_data = read_data_from_file(filename, first_header_line, second_header_line)

    # compute the length of the file in minutes
    maxDateTime = current_user_data[TIME_STAMP_COLUMN].max()
    TimeByFileInMinutes = (maxDateTime / 60000)
    print("The length of file %s est %s" % (filename, str(TimeByFileInMinutes)))

    # get the existing events and the missing events in the current file
    existing_events, missing_events = get_existing_missing_events(filename)

    file_missing_events = False
    if (len(missing_events) > 0):
        file_missing_events = True
    # split data by event (capteur) + resample the data on 128Hz
    dataByEvent = getDataByEvent(current_user_data, existing_events, TimeByFileInMinutes)
    # concat all sampled events in one table
    all_sampled_data = pd.concat(dataByEvent, axis=1)

    # read information related to competence value
    # there is two competence values for each post, after consulting sheet and after finishing the assembly (end of post)
    is_both_annotation_events = True
    if current_sequence_number == 'fin':
        # in fin file, user only give competence value by consulting sheet (he did not do the work asked in post)
        events = [Annotation]
        is_both_annotation_events = False
    else:
        events = [Annotation, Manual_Annotation]

    if is_an_exception_manual_annotation:
        events = [Exception_Manual_Annotation, Manual_Annotation]

    picklePrefix = current_user_name + '_' + current_sequence_number

    pickleName = picklePrefix + '_annotation_data'
    pickleFullPath = export_path + pickleName
    try:
        annotation_data = pd.read_pickle(pickleFullPath)
    except (OSError, IOError) as e:
        # filter lines of annotations
        print("filter")
        annotation_data = getAnnotationData(current_user_data, events)

        with open(pickleFullPath, 'wb') as annotation_data_file:
            pickle.dump(annotation_data, annotation_data_file)

    pickleName = picklePrefix + '_annotation_dict'
    pickleFullPath = export_path + pickleName
    try:
        annotation_dict = pd.read_pickle(pickleFullPath)
    except (OSError, IOError) as e:
        # extract the information of both annotations for each post
        annotation_dict = extractAnnotationData(annotation_data, is_both_annotation_events,
                                                is_an_exception_manual_annotation)

        with open(pickleFullPath, 'wb') as annotation_dict_file:
            pickle.dump(annotation_dict, annotation_dict_file)

    pickleName = picklePrefix + 'full_features_10Classes_data.singleF_array'
    pickleFullPath = export_path + pickleName
    try:
        all_data_with_class_for_current_file = pd.read_pickle(pickleFullPath)
    except (OSError, IOError) as e:
        all_data_with_class_for_current_file = listDatabyEachAnnotationValue(annotation_dict, all_sampled_data,
                                                                             classes_numbers=10,
                                                                             is_both_annotation_events=is_both_annotation_events)
        # np.savetxt("/home/deep01/zaher/export/"+current_user_name+"_"+current_sequence_number+"_data_list.csv", all_data_with_class_for_current_file, delimiter=",")
        with open(export_path + current_user_name + '_' + current_sequence_number + '_full_features_data.array',
                  'wb') as data_current_file:
            pickle.dump(all_data_with_class_for_current_file, data_current_file)

    if normalise_for_one_file:
        X_train, X_test, y_train, y_test = split_train_test_data(all_data_with_class_for_current_file)
        normalized_train_X, normalized_test_X = normalize(X_train, X_test)
        return normalized_train_X, normalized_test_X, y_train, y_test
    elif split_by_file:
        X_train, X_test, y_train, y_test = split_train_test_data(all_data_with_class_for_current_file)
        return X_train, X_test, y_train, y_test
    else:
        return all_data_with_class_for_current_file



def runAllFilesInRepository(csv_repository_path, export_path)
    full_data_list = []

    # get all files in repository
    listOfFiles = os.listdir(csv_repository_path)
    pattern = "*.csv"
    
    # loop each file
    for entry in listOfFiles:
        # filter csv file
        if fnmatch.fnmatch(entry, pattern):
            filename = csv_repository_path + "/" + entry
            print(filename)
            # run the file
            sequence_list = runOneFile(filename, export_path, False, False, False)
            full_data_list = np.concatenate((full_data_list, sequence_list))
    # export data of all file to an array (pickle)
    with open('new_export/all_data_full_features_10_classes.array', 'wb') as all_data_file:
        pickle.dump(full_data_list, all_data_file)
    # uncomment next line, if you need to export the pre-processed data to a csv file
    # np.savetxt("/home/deep01/zaher/export/full_data_list.csv", full_data_list, delimiter=",")



export_path = '/home/deep01/zaher/new_export/exportedObjectFullFeatures/'
# filename="/home/deep01/imotions_data/003_3754b5_sequence2.csv"
# sequence_list = []
# sequence_list = runOneFile(filename,export_path, False, False, False)


path = '/home/deep01/imotions_data_juin'
runAllFilesInRepository(path, export_path)
