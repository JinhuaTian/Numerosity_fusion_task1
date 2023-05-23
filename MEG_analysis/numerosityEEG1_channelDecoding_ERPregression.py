#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:13:38 2021

@author: tianjinhua
"""

'''
# plot psd
# select the largest amplitude according to psd plot
'''
from unittest import result
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
# import matplotlib
# matplotlib.use('Qt5Agg') #TkAgg
#from libsvm import svmutil as sv # pip install -U libsvm-official
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
# subjid = 'subj022' #subjName = ['subj016']
subjList = ['subj002','subj003']
#eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/stimuli/ModelRDM_NumIsTfaDenLLF.npy',encoding='unicode_escape',dtype=int)
eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/STI2.txt')
newSamplingRate = 500
repeat = 100
kfold = 3
labelNum = 45
eventNum = 9
eventName = ['Num1IA1','Num1IA2','Num1IA3','Num2IA1','Num2IA2','Num2IA3','Num3IA1','Num3IA2','Num3IA3']
tpoints = int(newSamplingRate*0.8) #-0.1~0.7ms
session = 12
decodingNum = 2 #'num', 'ia'
numList = []
for i in range(1, 17):
    numList.append(np.arange(((i-1)*5+1),((i-1)*5+6)))
numList = np.array(numList)

# transfer the event label 1~45 to 1~9 
def transLabel(events):
    for i in range(1,labelNum+1): #do not forget the 45 
        events[events[:,2]==i,2] = eventMatrix[eventMatrix[:,2]==i,3]
    return events

for subjid in subjList:
    # compute MEG RDM using pairwise SVM classification
    print('subject ' + subjid +' is running.')
    savePath = pj(rootDir, subjid, 'preprocessed')
    
    # epoch and label slice
    epochList1,labelList1,epochList2,labelList2=[],[],[],[] # data slice and corresponding label
    
    epochCount1,epochCount2 = 0,0
    # load data
    epochs_list1, epochs_list2 = [],[] # real data
    # walk through subj path, concatenate single subject's data to one file
    # search for epochList(epoch number) and labelList (label number)
    for sess in range(session):
        fifpath1 = pj(savePath, 'num'+str(sess)+'.fif')
        epoch1 = mne.read_epochs(fifpath1, preload=True, verbose=True)

        fifpath2 = pj(savePath, 'ia'+str(sess)+'.fif')
        epoch2 = mne.read_epochs(fifpath2, preload=True, verbose=True)

        # select label array
        labelList1.append(epoch1.events[:, 2])        
        epochData1 = epoch1.get_data(picks = 'eeg')

        labelList2.append(epoch2.events[:, 2])        
        epochData2 = epoch2.get_data(picks = 'eeg')
        
        # select label, epoch number  
        nEpochs1, nChan, nTime = epochData1.shape
        nEpochs2, nChan, nTime = epochData2.shape

        epochList1.append(list(range(epochCount1,epochCount1+nEpochs1)))
        epochList2.append(list(range(epochCount2,epochCount2+nEpochs2)))
        
        epochCount1 = epochCount1 + nEpochs1
        epochCount2 = epochCount2 + nEpochs2
        
        epochs_list1.append(epoch1)
        epochs_list2.append(epoch2)
        del epoch1, epochData1,epoch2, epochData2

# downsample to 500Hz
epochs_all1 = mne.concatenate_epochs(epochs_list1)
epochs_all2 = mne.concatenate_epochs(epochs_list2)
epochs_all1.resample(
    sfreq=newSamplingRate,
    npad="auto",
    window="boxcar",
    # n_jobs=4,
    pad="edge",
    verbose=True)
epochs_all2.resample(
    sfreq=newSamplingRate,
    npad="auto",
    window="boxcar",
    # n_jobs=4,
    pad="edge",
    verbose=True)

def transPoing2time(data): # -0.2~0.8
    data = data*2 - 200
    return data

picks1 = ['C1', 'C2', 'CZ']
picks2 = ['PO3','PO4','POZ']
pointWindow = [158,188,194,236] # 0.116，0.176，0.186，0.27 + 0.2sf
peakName = ['N1','P1','N2','P3']
pointRange = [[148,168],[178,198],[184,204],[226,246]]
PoNe = ['-','+','-','+'] # np.min or np.max

peakValues = np.zeros((2,labelNum,len(peakName))) # 2 data, 45 label number , 4 component
peakLatencies = np.zeros((2,labelNum,len(peakName)))

evokedList1,evokedList2 = [],[]
for i in range(1,labelNum+1):
    evokedList1.append(epochs_all1[str(i)].average())
    evokedList2.append(epochs_all2[str(i)].average())
evokedAll = [evokedList1,evokedList2]

for i in range(2): # two task
    for j in range(labelNum): # 45 image labels
        for m in range(len(peakName)):        
            data = evokedAll[i][j].get_data(picks=picks1)
            data = np.average(data,axis=0)
            if PoNe[m] =='+':
                peak = np.argmax(data[pointRange[m][0]:pointRange[m][1]])
            elif PoNe[m] =='-':
                peak = np.argmin(data[pointRange[m][0]:pointRange[m][1]])
            # peakValue = np.average(data[pointRange[m][0]:pointRange[m][1]])
            peak = peak + pointRange[m][0]
            peakLatency = transPoing2time(peak)
            peakLatencies[i,j,m]=peakLatency
            # flexible peak value, peak +- 10ms
            peakValue = np.average(data[(peak-5):(peak+5)]) #+-10ms
            peakValues[i,j,m]=peakValue

# regression
import statsmodels.api as sm
factorX = eventMatrix[:,0:2]
for i in range(2):
    for m in range(len(peakName)):  
        # model = sm.OLS(peakLatencies[i,:,m], factorX) #生成模型
        model = sm.OLS(peakValues[i,:,m], factorX)
        result = model.fit() #模型拟合
        print(result.summary())
'''
# plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for m in range(labelNum):
    ax.scatter(int(eventMatrix[m,0]), int(eventMatrix[m,1]), peakLatencies[0,m,1],c='black')
plt.xticks([1,2,3])
plt.yticks([1,2,3])
plt.title('P1 latentcy')
ax.set_xlabel('Number')
ax.set_ylabel('Item area')
ax.set_zlabel('Latency(ms)')
plt.show()
'''


'''
epochs_all1.info['bads'].extend(['CB1','CB2'])
# epochs_all1.plot_psd()
Evoked_all1=epochs_all1.average()
Evoked_all1.plot()
dd1 = Evoked_all1.plot_topomap()
dd1.savefig('toponum.png')
'''

print('All Done')