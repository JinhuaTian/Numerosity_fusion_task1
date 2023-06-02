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
subjList = ['subj003']  #'subj002','subj003'
RDMtype = ['number','item area']
newSamplingRate = 500
# sessions(12) x train data x label x test data x label x tpoints
fullData=np.zeros((len(subjList),12,2,2,2,2,int(newSamplingRate*0.8)))
for i in range(len(subjList)):
    data = np.load(pj(rootDir, 'crossDecoding12x'+str(newSamplingRate)+'hz_'+ subjList[i] +'.npy'))
    fullData[i] = data
print('initing')
avgdata = np.average(fullData,axis=(0,1))
#plot the data
fig = plt.figure(figsize=(9, 6), dpi=100)

plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,0,0,:],color='r',label='Number decoding accuracy(number task)') # color='r',color='b',
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,1,0,1,:],color='b',label='Item area decoding accuracy(number task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,0,1,0,:],color='r',linestyle="--",label='Number decoding accuracy(Item area task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,1,1,:],color='b',linestyle="--",label='Item area decoding accuracy(Item area task)')
plt.axhline(y=0.333,xmin=-100,xmax= 700,color='black',linestyle="--")
# plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
plt.xlabel('Time points(ms)')
plt.ylabel('Decoding accuracy(%)')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
plt.savefig(pj(rootDir, 'CorssDecoding'+'3'+'.png'))
plt.show()

'''
fig = plt.figure(figsize=(9, 6), dpi=100)

plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,0,0,:],color='r',label='Within magnitude decoding accuracy(number task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,1,1,:],color='b',label='Cross magnitude decoding accuracy(number task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,1,1,:],color='b',linestyle="--",label='Within magnitude decoding accuracy(Item area task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,0,0,:],color='r',linestyle="--",label='Cross magnitude decoding accuracy(Item area task)')
plt.axhline(y=0.333,xmin=-100,xmax= 700,color='black',linestyle="--")
# plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
plt.xlabel('Time points(ms)')
plt.ylabel('Decoding accuracy(%)')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
plt.savefig(pj(rootDir, 'CorssDecoding_wc'+'3'+'.png'))
plt.show()
'''
color = ["Red", "Purple", "Gray", "Blue", "Green", "Orange", 'brown']
fig = plt.figure(figsize=(9, 6), dpi=100)

plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,0,0,:],color=color[0],label='Train on number test on number (number task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,1,1,:],color=color[1],label='Train on number test on IA (number task & IA task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,1,0,:],color=color[2],label='Train on number test on number (number task & IA task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,1,1,:],color=color[3],linestyle="--",label='Train on IA test on IA (Item area task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,0,0,:],color=color[4],linestyle="--",label='Train on IA test on number (IA task & number task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,0,1,:],color=color[5],linestyle="--",label='Train on IA test on IA (IA task & number task)')
plt.axhline(y=0.333,xmin=-100,xmax= 700,color='black',linestyle="--")
# plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
plt.xlabel('Time points(ms)')
plt.ylabel('Decoding accuracy(%)')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
plt.savefig(pj(rootDir, 'CorssDecoding_wc'+'3'+'.png'))
plt.show()

print('Done')
print('Done')