# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:22:51 2021

@author: tclem
"""
'''
Data segmentation
'''
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # use LDA to classify data
import pandas as pd

scaler = StandardScaler()

from numba import jit
jit(nopython=True, parallel=True)

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
subjid = 'subj003'  # subjName = ['subj016']
# 'subj002','subj003','subj004','subj005','subj006','subj007','subj008','subj009','subj010'
trialNum = 12
savePath = pj(rootDir, subjid, 'preprocessed')
dictMatrix = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
        '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21,
        '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30, '31': 31, '32': 32,
        '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40, '41': 41, '42': 42, '43': 43,
        '44': 44, '45': 45, '700001': 98, '71': 71, '72': 72, '49': 49, '50': 50, '51': 51, '99': 99}
for file in os.listdir(savePath):
    if 'filteredRefICA1.set' in file:
        path = pj(savePath,file)
        # read data and annotations
        data = mne.io.read_raw_eeglab(path)
        # modify data
        channelName = data.info.ch_names
        channelDict = {channelName[i]:"eeg" for i in range(62)}
        channelDict = {'HEO':"eog", 'VEO':"eog", 'TRIGGER':'stim'}
        data.set_channel_types(channelDict)
        # read annotations
        events = mne.events_from_annotations(data,event_id=dictMatrix)
        events = events[0]
        #
        indexMatrix1 = []
        indexMatrix2 = []

        index1 = np.where(events[:,2]==71)[0]
        index2 = np.where(events[:,2]==72)[0]
        index1 = pd.DataFrame(index1,columns=['data'])
        index1.insert(index1.shape[1],'type',1)
        index2 = pd.DataFrame(index2,columns=['data'])
        index2.insert(index2.shape[1],'type',2)
        index = pd.concat([index1,index2],axis=0)
        index = index.sort_values(by=['data'])
        for i in range(index.shape[0]):
            if i != (index.shape[0]-1):
                interval = events[index.iloc[i,0]:index.iloc[i+1,0],:] #,closed='neither'
            elif i == (index.shape[0]-1): # append the last events
                interval = events[index.iloc[i,0]:,:]
            if index.iloc[i,1]==1:
                indexMatrix1.append(interval)
            elif index.iloc[i,1]==2:
                indexMatrix2.append(interval)
        
        events1 = np.concatenate(np.array(indexMatrix1))
        events2 = np.concatenate(np.array(indexMatrix2))
        reject = dict(eeg=2e-4)      #unit: V (EEG channels), unit: V (EOG channels)
        # save each session file:
        for i in range(trialNum):
            session1 = mne.pick_events(indexMatrix1[i], exclude=[49,50,51,71,72,98,99])
            sessData1 = mne.Epochs(data, events=session1, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
            sessData1.apply_baseline((-0.2, 0))
            path1 = pj(savePath,'num'+str(i)+'.fif')
            sessData1.save(path1, overwrite=True)

            session2 = mne.pick_events(indexMatrix2[i], exclude=[49,50,51,71,72,98,99])
            sessData2 = mne.Epochs(data, events=session2, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
            sessData2.apply_baseline((-0.2, 0))
            path2 = pj(savePath,'ia'+str(i)+'.fif')
            sessData2.save(path2, overwrite=True)
        # save all files
        trial1 = mne.pick_events(events1, exclude=[49,50,51,71,72,98,99])
        trial2 = mne.pick_events(events2, exclude=[49,50,51,71,72,98,99])

        rej1 = mne.Epochs(data, events=trial1, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
        rej2 = mne.Epochs(data, events=trial2, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
        rej1.apply_baseline((-0.2, 0))
        path1 = pj(savePath,'num.fif')
        rej1.save(path1, overwrite=True)
        rej2.apply_baseline((-0.2, 0))
        path2 = pj(savePath,'ia.fif')
        rej2.save(path2, overwrite=True)




