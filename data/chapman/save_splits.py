"""
Adapated from @danikiyasseh 
Source: https://github.com/danikiyasseh/loading-physiological-data/blob/master/load_chapman_ecg.py
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

enc = LabelEncoder()
dataset = 'chapman'
basepath = './raw'

files = os.listdir(os.path.join(basepath,'ECGDataDenoised'))
database = pd.read_csv(os.path.join(basepath,'Diagnostics.csv'))
dates = database['FileName'].str.split('_',expand=True).iloc[:,1]
dates.name = 'Dates'
dates = pd.to_datetime(dates)
database_with_dates = pd.concat((database,dates),1)

# Modify output label as "" (arrhythymia classification), "gender" (gender classification), "age" (age regression)
task = "gender"
options = {"age": "PatientAge", "gender": "Gender", "": "Rhythm"}

""" Combine Rhythm Labels """
old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
database_with_dates['Rhythm'] = database_with_dates['Rhythm'].replace(old_rhythms,new_rhythms)
# unique_labels = database_with_dates['Rhythm'].value_counts().index.tolist()
unique_labels = database_with_dates['Gender'].value_counts().index.tolist()
enc.fit(unique_labels)

""" Combine Dates """
def combine_dates(date):
    new_dates = ['All Terms']
    cutoff_dates = ['2019-01-01']
    cutoff_dates = [pd.Timestamp(date) for date in cutoff_dates]
    for t,cutoff_date in enumerate(cutoff_dates):
        if date < cutoff_date:
            new_date = new_dates[t]
            break
    return new_date
database_with_dates['Dates'] = database_with_dates['Dates'].apply(combine_dates)

""" Patients in Each Task and Phase """
phases = ['train','val','test']
phase_fractions = [0.8, 0.1, 0.1]
seed = 42
phase_fractions_dict = dict(zip(phases,phase_fractions))
term = 'All Terms'
reduce_dataset = 0.02 # fraction of dataset to create

term_phase_patients = dict()
term_patients = database_with_dates['FileName'][database_with_dates['Dates'] == "All Terms"]
random_term_patients = term_patients.sample(frac=1,random_state=seed)
start = 0
for phase,fraction in phase_fractions_dict.items():
    if phase == 'test':
        phase_patients = random_term_patients.iloc[start:].tolist() #to avoid missing last patient due to rounding
    else:
        npatients = int(reduce_dataset*fraction*len(term_patients))
        phase_patients = random_term_patients.iloc[start:start+npatients].tolist()
    term_phase_patients[phase] = phase_patients
    start += npatients

sampling_rate = 500
modality_list = ['ecg']
fraction_list = [1]
leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
desired_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
inputs_dict = dict()
outputs_dict = dict()
pids = dict()


for phase in phases:
    current_patients = term_phase_patients[phase]
    current_inputs = []
    current_outputs = []
    current_pids = []
    for patient in tqdm(current_patients):
        filename = patient + '.csv'
        data = pd.read_csv(os.path.join(basepath,'ECGDataDenoised',filename)) #SxL
        
        resampling_length = 5000
        data_resampled = resample(data,resampling_length)
        data_resampled = data_resampled.T #12x2500
        lead_indices = np.where(np.in1d(leads,desired_leads))[0]
        data_resampled = data_resampled[lead_indices,:] #12x2500
        
        label = database_with_dates[options[task]][database_with_dates['FileName']==patient]
        if task in ["", "gender"]:
            label = enc.transform(label).item()
        
        current_inputs.append(data_resampled)
        current_outputs.append([label for _ in range(data_resampled.shape[0])])
        current_pids.append([patient for _ in range(data_resampled.shape[0])])

    inputs_dict[phase] = np.array(current_inputs)
    outputs_dict[phase] = np.array(current_outputs)
    pids[phase] = np.array(current_pids)


""" Less Computationally Intensive NaN Removal Process """
bad_indices = dict()
for phase in phases:
    bad_indices[phase] = []
    for row,entry in enumerate(inputs_dict[phase]):
        if np.isnan(entry).sum() > 0:
            bad_indices[phase].append(row)
    """ Delete NaN Rows """
    inputs_dict[phase] = np.delete(inputs_dict[phase],bad_indices[phase],0)
    outputs_dict[phase]= np.delete(outputs_dict[phase],bad_indices[phase],0)
    pids[phase] = np.delete(pids[phase],bad_indices[phase],0)
    
def save_final_frames_and_labels(frames_dict,labels_dict,save_path):
    frames_dict['train'].dump(os.path.join(save_path, "{}X_train.npy".format(reduce_dataset)), protocol=4)
    frames_dict['val'].dump(os.path.join(save_path, "{}X_val.npy".format(reduce_dataset)), protocol=4)
    if 'test' in phases: frames_dict['test'].dump(os.path.join(save_path, "{}X_test.npy".format(reduce_dataset)), protocol=4)

    labels_dict['train'][:,0].dump(os.path.join(save_path, "{}y_train.npy".format(reduce_dataset)), protocol=4)
    labels_dict['val'][:,0].dump(os.path.join(save_path, "{}y_val.npy".format(reduce_dataset)), protocol=4)
    if 'test' in phases: labels_dict['test'][:,0].dump(os.path.join(save_path, "{}y_test.npy".format(reduce_dataset)), protocol=4)

save_path = "./seed{}{}".format(seed, task)
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_final_frames_and_labels(inputs_dict,outputs_dict,save_path)