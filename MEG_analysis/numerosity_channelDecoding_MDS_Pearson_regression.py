# -*- coding: utf-8 -*-
'''
prepare data as n subjects x 45 events&2 tasks x n times
nomorlize data? loss
compute pearson correlation for each time points 1-r
'''
import numpy as np
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from os.path import join as pj

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
# subjid = 'subj022' #subjName = ['subj016']
subjList = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025']  
# 'subj001','subj003','subj004','subj006','subj008','subj009','subj010'

newSamplingRate = 500
labelNum = 45

tpoints = int(newSamplingRate*0.7)+1 #-0~0.6ms
tmin=0
tmax=0.7

session = 14
tasks = ['num','fa']

method = 'pearson' #'pearson' 'euclidean' 'mahalanobis' (sample>dimension)

data = np.zeros((len(subjList),labelNum*2,273,tpoints))
subjCount = 0
for subjid in subjList:
    # compute MEG RDM using pairwise SVM classification
    print('subject ' + subjid +' is running.')
    savePath = pj(rootDir, subjid, 'preprocessed')
    
    labelCount = 0
    # walk through subj path, concatenate single subject's data to one file
    # search for epochList(epoch number) and labelList (label number)
    for task in tasks:
        epochs_list = []
        for sess in range(session):
            fifpath = pj(savePath, task+str(sess+1)+'.fif')
            epoch = mne.read_epochs(fifpath, preload=True, verbose=True)
            epochs_list.append(epoch)
            del epoch
        epochs_all = mne.concatenate_epochs(epochs_list)
        epochs_all.resample(
            sfreq=newSamplingRate,
            npad="auto",
            window="boxcar",
            # n_jobs=4,
            pad="edge",
            verbose=True)
        del epochs_list

        # Count channel number
        # channelType = [['MLO','MRO','MZO'],['MLP','MRP','MZP'],['MLF','MRF','MZF']]
        # picksAll = []
        # for channels in channelType:
        #     picksTmp = []
        #     for part in channels:
        #         picks = mne.pick_channels_regexp(epochs_all.ch_names, part)
        #         picksTmp = picksTmp + picks
        #     print(len(picksTmp))
        #     picksAll.append(picksTmp)
                

        for i in range(1,labelNum+1):
            epochDat = epochs_all[str(i)].average().crop(tmin=tmin,tmax=tmax,include_tmax=True)
            # normalize data
            epochDat = scaler.fit_transform(epochDat.get_data(picks = 'mag'))
            
            data[subjCount,labelCount,:,:] = epochDat
            labelCount = labelCount+1
        
        # epochDat.plot()
        
    subjCount = subjCount+1
    print('subject Done')

# compute pair number:
vectorLength = 0
# calculate label pairs
labelPair = np.array([],dtype=int)
# should be 25 * 2 = 50 labels
for x in range(labelNum*2):
    for y in range(x+1,labelNum*2):
        labelPair = np.hstack((labelPair,[x,y]))
        vectorLength = vectorLength + 1

corrData = np.zeros((len(subjList),tpoints,vectorLength))

from scipy.stats import pearsonr
for subj in range(len(subjList)):
    for tp in range(tpoints):
        count = 0
        for x in range(labelNum*2):
            for y in range(x+1,labelNum*2):
                if method == 'pearson':  # 1-pearson OR pearson?
                    corrData[subj,tp,count],_ = pearsonr(data[subj,x,:,tp],data[subj,y,:,tp])
                elif method == 'euclidean':
                    corrData[subj,tp,count] = np.linalg.norm(data[subj,x,:,tp]-data[subj,y,:,tp])
                elif method == 'mahalanobis':
                    print('how?')#corrData[subj,tp,count] = 
                count = count+1

# dissimilarity
if method == 'pearson':
    corrData = 1 - corrData

np.save(pj(rootDir, "MDS_"+ method +".npy"), corrData)

print('All done')