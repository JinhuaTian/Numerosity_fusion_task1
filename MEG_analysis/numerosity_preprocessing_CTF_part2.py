'''
Script for MEG data preprocessing
# preprocessing procedure:
1.load data
2.run SSP, head motion correction (to the first run)
3.filter:notch, high pass, low pass
4.run ICA  # before epoch, 1 Hz highpass suggested
5.epoch data (auto rejection, baseline correction)

# file2:
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
from mne.io import read_raw_fif

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
subjName =  ['subj025'] ## 'subj001''subj002','subj003','subj004','subj005'
taskName = 'raw'
### ICA compo: 9 14 56 62
# filter parameters
reject = dict(mag=4e-12) #eog=250e-6, there is no EOG!
manualInput = True

from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)
for subj in subjName:
    # rawDir = pj(rootDir, subj, taskName)
    # name and makedir the save path
    savePath = pj(rootDir, subj, 'preprocessed')
    # load filtered data
    for i in range(1,16): #range(1,13): # 1~15, 14 in total
        fileName = 'filterEpochTemp_' + str(i) + '.fif'
        filepath = pj(savePath,fileName)
        raw = read_raw_fif(filepath, preload=True) #allow_maxshield=True,
        # check the channel quality
        if i == 1:
            raw.plot(block = True)
        # --------------------------------------
        # 1.2 run ICA and reject artifact components
        # just remove eye blink, horizontal eye movement and muscle
        # --------------------------------------
        from mne.preprocessing import ICA  #, create_eog_epochs, create_ecg_epochs,corrmap
        ica = ICA() # n_components=90, 
        ica.fit(raw) # no need to set picks, fitting ICA to 273 channels
        ica.plot_sources(raw, show_scrollbars=True) #,block=True
        ica.plot_components() #0 14 17 21 32; 0 1 6; 2 10 32

        #reject ica components from input or from click
        if manualInput == True:
            ica_rej = input()
            print(ica_rej)
            bad_comps = ica_rej.split(" ")
            bad_comps = [int(bad_comps[i]) for i in range(len(bad_comps))] #transform str to number
            #ica.exclude = list(ica_rej)
            ica_rej = ica.apply(raw,exclude=bad_comps)
        elif manualInput == False:
            ica_rej = ica.apply(raw)
        #plot psd 
        ica_rej.plot_psd(fmax=400)
        # --------------------------------------
        # 1.3 Epoch and reject bad epochs
        # --------------------------------------
        # peak-to-peak amplitude rejection parameters
        # select events and epoch data
        events = mne.find_events(ica_rej, stim_channel='UPPT001', shortest_event=2,min_duration=0.005)
        index1 = np.where(events[:,2]==71)[0]
        index2 = np.where(events[:,2]==72)[0]
        index = np.append(index1,index2)
        index=np.append(index,events.shape[0])        
        NumList = []
        FaList = []
        index.sort() # sort the index
        for jj in range(index.shape[0]):
            if index1.__contains__(index[jj]):
                NumList.append(events[index[jj]:index[jj+1],:])
            elif index2.__contains__(index[jj]):
                FaList.append(events[index[jj]:index[jj+1],:])
        NumList = np.concatenate(NumList)
        FaList = np.concatenate(FaList)
        session1 = mne.pick_events(NumList, exclude=[49,51,59,61,71,72])
        sessData1 = mne.Epochs(ica_rej, events=session1, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
        sessData1.apply_baseline((-0.2, 0))
        path1 = pj(savePath,'num'+str(i)+'.fif')
        sessData1.save(path1, overwrite=True)

        session2 = mne.pick_events(FaList, exclude=[49,51,59,61,71,72])
        sessData2 = mne.Epochs(ica_rej, events=session2, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject, preload=True, detrend=1, verbose=True)
        sessData2.apply_baseline((-0.2, 0))
        path2 = pj(savePath,'fa'+str(i)+'.fif')
        sessData2.save(path2, overwrite=True)
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
        del raw, ica_rej
print('All Done')