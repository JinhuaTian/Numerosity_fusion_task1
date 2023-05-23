#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib.use('Qt5Agg') #TkAgg
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
# subjid = 'subj022' #subjName = ['subj016']
# 'subj009','subj011','subj012','subj013','subj017','subj018','subj020','subj021','subj022','subj023'
subjList = ['subj002','subj003']  #'subj002','subj003'
RDMtype = ['number','item area']
newSamplingRate = 500
# sessions(12) x train data x label x test data x label x tpoints
fullData=np.zeros((len(subjList),12,2,2,2,2,int(newSamplingRate*0.8),int(newSamplingRate*0.8)))
for i in range(len(subjList)):
    data = np.load(pj(rootDir, 'crossDecoding_CD22TG_12x2x2'+str(newSamplingRate)+'hz_'+ subjList[i] +'.npy'))
    fullData[i] = data
avgData = np.average(fullData,axis=(0,1)) # 2x2x2x2xtpsxtps
#v axis is training data, h axis is testing data
fig1 =plt.figure(figsize=(20,20),dpi=300)
count = 1
name = ['Num','IA']
for i in range(2):
    for j in range(2):
        for m in range(2):
            for n in range(2):
                plt.subplot(4,4,count)
                sns.heatmap(np.flip(avgData[i,j,m,n,:,:],axis=0),cmap='jet',vmin=0.3, vmax=0.4,cbar=True) #*1000/newSamplingRate
                plt.ylabel('Train on '+name[i]+' data ('+name[j]+' label)')
                plt.xlabel('Test on '+name[m]+' data ('+name[n]+' label)')
                count = count +1
fig1.savefig('matplot.png')
plt.show()

print('initing')