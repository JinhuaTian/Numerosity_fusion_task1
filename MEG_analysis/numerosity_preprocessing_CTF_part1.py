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
import os
'''
import sys
sys.path.append('/data/home/nummag01/workingdir/pythonPackage/mne/')
sys.path.remove('/data/home/nummag01/.local/lib/python3.7/site-packages/mne/')
'''
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
dataDir = '/data/user/swap_zk/Magnitude/fusion_task1/MEG'

rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
'''
'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj021','subj023','subj024','subj025',
 'subj028','subj029','subj031','subj033','subj034','subj037','subj038','subj040'
'''
subjName = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018']
# 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008'
taskName = 'raw'
badChannels=['BG1-4504', 'BG2-4504', 'BG3-4504', 'BP1-4504', 'BP2-4504', 'BP3-4504', 'BR1-4504', 'BR2-4504', 'BR3-4504', 'G11-4504', 'G12-4504', 
'G13-4504', 'G22-4504', 'G23-4504', 'P11-4504', 'P12-4504', 'HLC0037-4504', 'HLC0036-4504', 'HLC0035-4504', 'HLC0034-4504', 'HLC0027-4504', 
'HLC0025-4504', 'HLC0024-4504', 'HLC0017-4504', 'HLC0016-4504', 'HLC0015-4504', 'HLC0014-4504', 'HLC0026-4504', 'MPLLU710', 'MRSYN710', 'MSTAT710', 
'MMSTC709', 'MMSTC710', 'MPLLU709', 'MSTAT709', 'MRSYN709', 'MMSTC708', 'MPLLU708', 'MRSYN708', 'MSTAT708', 'MMSTC707', 'MRSYN707', 'MSTAT707', 
'MMSTC706', 'MPLLU706', 'MSTAT706', 'MMSTC705', 'MPLLU705', 'MRSYN705', 'MSTAT705', 'MMSTC704', 'MPLLU704', 'MRSYN704', 'MSTAT704', 'MRSYN706', 
'MPLLU707', 'MMSTC703', 'MPLLU703', 'MRSYN703', 'MSTAT703', 'MMSTC701', 'MPLLU701', 'MRSYN701', 'MSTAT701', 'MMSTC110', 'MRSYN110', 'MPLLU110', 
'MSTAT110', 'MPLLU109', 'MRSYN109', 'MSTAT109', 'MMSTC108', 'MPLLU108', 'MRSYN108', 'MSTAT108', 'MMSTC109', 'MMSTC107', 'MPLLU107', 'MRSYN107', 
'MSTAT107', 'MMSTC106', 'MPLLU106', 'MRSYN106', 'MMSTC105', 'MPLLU105', 'MRSYN105', 'MSTAT105', 'MMSTC104', 'MPLLU104', 'MSTAT106', 'MRSYN104', 
'MSTAT104', 'MMSTC103', 'MPLLU103', 'MRSYN103', 'MSTAT103', 'HLC0011-4504', 'HLC0013-4504', 'HLC0022-4504', 'HLC0031-4504', 'HLC0032-4504', 
'HLC0018-4504', 'HLC0028-4504', 'HLC0038-4504', 'MSTAT101', 'MRSYN101', 'MPLLU101', 'MMSTC101', 'MSTAT102', 'MRSYN102', 'MPLLU102', 'MMSTC102', 
'HLC0023-4504', 'HLC0021-4504', 'HLC0012-4504', 'HLC0033-4504', 'HADC001-4504', 'HADC002-4504', 'HADC003-4504', 'HDAC001-4504', 'HDAC002-4504', 
'HDAC003-4504', 'SCLK01-177', 'R22-4504', 'R13-4504', 'R12-4504', 'R11-4504', 'Q23-4504', 'Q22-4504', 'Q13-4504', 'Q12-4504', 'Q11-4504', 
'P23-4504', 'P22-4504', 'P13-4504'] # 'UPPT001'

'''
# MEG channels
['BG1-4504', 'BG2-4504', 'BG3-4504', 'BP1-4504', 'BP2-4504', 'BP3-4504', 'BR1-4504', 'BR2-4504', 'BR3-4504', 'G11-4504', 
'G12-4504', 'G13-4504', 'G22-4504', 'G23-4504', 'P11-4504', 'P12-4504', 'R22-4504', 'R13-4504', 'R12-4504', 'R11-4504', 'Q23-4504', 
'Q22-4504', 'Q13-4504', 'Q12-4504', 'Q11-4504', 'P23-4504', 'P22-4504', 'P13-4504'] # 
'''

# filter parameters
freqs = np.arange(50, 200, 50)
highcutoff = 1 # A high-pass filter with 1 Hz cutoff frequency is recommended for ICA.
lowcutoff = 100 # 80Hz?
newSamplingRate = 500
#reject = dict(mag=3e-12) # 3000 fT eog=250e-6, there is no EOG!
manualInput = True

from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)
for subj in subjName:
    fileCount = 1 # i and count should match
    rawDir = pj(dataDir, subj, taskName)
    # name and makedir the save path
    savePath = pj(rootDir, subj, 'preprocessed')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    '''
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
    '''
    '''
    # load empty room data:
    #https://mne.tools/1.0/auto_tutorials/preprocessing/50_artifact_correction_ssp.html?highlight=empty%20room#creating-the-empty-room-projectors
    subjDir = pj(rootDir,subj)
    for file in os.listdir(subjDir): # walk through folder and search for Number i file
            if 'Noise-default' in file:
                noiseFile = pj(subjDir,file)
                #extended_proj = mne.preprocessing.maxwell_filter_prepare_emptyroom(noiseFile)
                empty_room_raw = read_raw_ctf(noiseFile)

                # delete system projection  
                # empty_room_raw.del_proj()

                # plot empty room info
                for average in (False, True):
                    empty_room_raw.plot_psd(average=average, dB=False, xscale='log')
                #empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_grad=3, n_mag=3)
                #empty_room_projs = mne.preprocessing.maxwell_filter_prepare_emptyroom(empty_room_raw)
    '''
    for i in range(1,16): # 14 in total
        for file in os.listdir(rawDir): # walk through folder and search for Number i file
            if 'G15BNU' in file and '_{:02d}.ds'.format(i) in file:
                # save name and save path
                fileName = 'filterEpochTemp_' + str(fileCount) + '.fif'
                savePath2 = pj(savePath,fileName)

                filepath = pj(rawDir,file)
                raw = read_raw_ctf(filepath, preload=True) #allow_maxshield=True,
                # mark bad channels
                raw.info["bads"].extend(badChannels)

                '''
                # prepare empty proj for maxwellfiltering
                empty_room_projs = mne.preprocessing.maxwell_filter_prepare_emptyroom(raw_er=empty_room_raw,raw=raw)
                empty_room_projs = mne.compute_proj_raw(empty_room_projs, n_grad=0, n_mag=3)
                '''

                # take the first run as the reference run
                fname_head = pj(savePath,'headPos.pos') 
                if fileCount == 1: # take the first run as the reference run
                    dev_head_t_ref = raw.info['dev_head_t']
                    # save head position
                    with open(fname_head, 'wb') as f:
                        joblib.dump(dev_head_t_ref, f)
                    f.close()
                    # check the channel quality
                    # raw.plot(block = True)
                else:
                    dev_head_t_ref = joblib.load(fname_head)
                '''
                # plot the first run and check its quanlity
                if fileCount == 1: 
                    # inspect raw data
                    raw.plot(block = True)
                    raw.plot_psd(fmax=400)
                '''
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

                # mark and drop bad channels
                # raw.info["bads"].extend(badChannels)
                raw.drop_channels(badChannels)

                '''
                # plot raw after rough preprocessing
                if fileCount == 1: 
                    # inspect raw data
                    raw.plot(block = True)
                    raw.plot_psd(fmax=400)
                
                '''
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