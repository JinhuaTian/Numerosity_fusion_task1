'''
Script for MEG data preprocessing
# preprocessing procedure:
1.load data
2.run SSP, head motion correction (to the first run)
3.filter:notch, high pass, low pass

# Part2:
4.run ICA  # before epoch, 1 Hz highpass suggested
5.epoch data (auto rejection, baseline correction)

6.Annotate and delete bad epochs manually

# post prep
7.average epochs
8.concatenate data at subject level
9.source reconstruction
10.decoding
'''
import numpy as np
import os,sys
from os.path import join as pj
import time
from mne.io import read_raw_ctf

import matplotlib
matplotlib.use('TkAgg') #   Qt5Agg #'TkAgg'
'''
oldMne = '/usr/local/neurosoft/anaconda3/lib/python3.8/site-packages/mne' 
sys.path.remove(oldMne)
currMne = '/nfs/s2/userhome/tianjinhua/workingdir/code'
sys.path.append(currMne)
'''
import mne
import joblib

rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
subjName = ['subj024','subj025']
# 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008'
taskName = 'raw'

# filter parameters
freqs = np.arange(50, 200, 50)
highcutoff = 1 # A high-pass filter with 1 Hz cutoff frequency is recommended for ICA.
lowcutoff = 100 # 80Hz?
newSamplingRate = 500
reject = dict(mag=4e-12) #eog=250e-6, there is no EOG!
manualInput = True

from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)
for subj in subjName:
    fileCount = 1 # i and count should match
    rawDir = pj(rootDir, subj, taskName)
    # name and makedir the save path
    savePath = pj(rootDir, subj, 'preprocessed')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # extract head position and average
    full_dev_head_t_ref = []
    fname_head = pj(savePath,'headPos.pos') 
    for i in range(1,16): # 14 in total
        for file in os.listdir(rawDir): # walk through folder and search for Number i file
            if 'G15BNU' in file and '_{:02d}.ds'.format(i) in file:
                filepath = pj(rawDir,file)
                raw = read_raw_ctf(filepath, preload=True) #allow_maxshield=True,
                full_dev_head_t_ref.append(raw.info['dev_head_t']['trans'])
    dev_head_t_ref = np.average(full_dev_head_t_ref,axis=0)
    raw.info['dev_head_t']['trans'] = dev_head_t_ref 
    dev_head_t_ref = raw.info['dev_head_t']
    # save head position
    with open(fname_head, 'wb') as f:
        joblib.dump(dev_head_t_ref, f)
    f.close()
    for i in range(1,16): # 14 in total
        for file in os.listdir(rawDir): # walk through folder and search for Number i file
            if 'G15BNU' in file and '_{:02d}.ds'.format(i) in file:
                # save name and save path
                fileName = 'filterEpochTemp_' + str(fileCount) + '.fif'
                savePath2 = pj(savePath,fileName)

                filepath = pj(rawDir,file)
                raw = read_raw_ctf(filepath, preload=True) #allow_maxshield=True,
                # take the first run as the reference run
                fname_head = pj(savePath,'headPos.pos') 
                if fileCount == 1: # take the first run as the reference run
                    # inspect raw data
                    raw.plot(block = True)
                    raw.plot_psd(fmax=400)
                # --------------------------------------
                # 1.1 load data, run SSP, filter
                # --------------------------------------
                raw.apply_gradient_compensation(0)  # must un-do software compensation first
                # origin=(0., 0., 0.04) for adults
                # origin=(0., 0., 0.) for subj002
                raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.04), coord_frame='head', destination=dev_head_t_ref) # origin=(0., 0., 0.),''auto',ignore_ref=True,

                # 2. filter the data: notch, high pass, low pass
                # meg_picks = mne.pick_types(raw.info, mag=True, eeg=False, eog=False)
                raw = raw.notch_filter(freqs=freqs, picks='mag', method='spectrum_fit', filter_length="auto", phase="zero", verbose=True) #n_jobs=4, 
                raw = raw.filter(l_freq=highcutoff, h_freq=None)
                raw = raw.filter(l_freq=None, h_freq=lowcutoff)
                #raw.plot(block = True)
                if fileCount == 1: # plot raw after preprocessing
                    # inspect raw data
                    raw.plot(block = True)
                    raw.plot_psd(fmax=400)
                raw.save(savePath2, overwrite=True)
                fileCount = fileCount + 1
                '''
                # --------------------------------------
                # 1.3 Reject epochs manually
                # --------------------------------------
                #select and annotate bad epoch
                fig = ica_rej.plot(picks='mag',block=True)
                #fig.canvas.key_press_event('a')

                #apply bad epoch
                ica_rej.drop_bad()
                # save the manual rejected file
                fileName = 'filterEpochICAMr_' + file
                tempSavename = pj(savePath,fileName)
                ica_rej.save(tempSavename, overwrite=True)
                '''
                del raw

print('All Done')