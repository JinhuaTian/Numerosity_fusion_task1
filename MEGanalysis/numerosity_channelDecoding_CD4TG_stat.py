#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
# matplotlib.use('Qt5Agg') #TkAgg
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
# 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
#  'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025'
subjList = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025']  # lack16!
RDMtype = ['number','Field area']
newSamplingRate = 200
# sessions(12) x train data x label x test data x label x tpoints
fullData=np.zeros((len(subjList),2,2,2,2,int(newSamplingRate*0.8),int(newSamplingRate*0.8)))
for i in range(len(subjList)): # 
    data = np.load(pj(rootDir, 'crossDecoding_CD22TG_svm12x2x2x'+str(newSamplingRate)+'hz_'+ subjList[i] +'.npy'))
    fullData[i] = data
avgData = np.average(fullData,axis=0) # 2x2x2x2xtpsxtps
#v axis is training data, h axis is testing data
fig1 =plt.figure(figsize=(20,20),dpi=100)
plt.title('Temporal generalization')
count = 1
xTicks = np.arange(-98,702,5)
yTicks = np.arange(700,-100,-5)#np.arange(-100,700,2)
locator = np.arange(0,800,100)
name = ['Num','FA']
for i in range(2):
    for j in range(2):
        for m in range(2):
            for n in range(2):
                ax = plt.subplot(4,4,count)
                tempData = np.flip(avgData[i,j,m,n,:,:],axis=0) # flip upside down 
                pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
                sns.heatmap(pd_data,cmap="jet",cbar=True,vmin=0.31, vmax=0.35) # ,vmin=0.2, vmax=0.5*1000/newSamplingRate,mask =mask,'jet'
                #plt.gca().xaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600])) 
                #plt.gca().yaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600]))           
                plt.ylabel('Train on '+name[i]+' task ('+name[j]+' label)',fontsize=20)
                plt.xlabel('Test on '+name[m]+' task ('+name[n]+' label)',fontsize=20)
                
                count = count +1
plt.tight_layout()
fig1.savefig('matplot_numfa.png')
plt.show()

# run a correlation analysis


# dpi = 100
# fig = plt.figure(figsize=(9, 6), dpi=dpi)
# tempData = np.flip(avgData[i,j,m,n,:,:],axis=0) # flip upside down 
# pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
# sns.heatmap(pd_data,cmap='jet',vmin=0, vmax=0.3,cbar=True) #*1000/newSamplingRate,could add mask, mask =mask
# #plt.gca().xaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600])) 
# #plt.gca().yaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600]))           
# plt.ylabel('Train on '+name[i]+' task ('+name[j]+' label)',fontsize=20)
# plt.xlabel('Test on '+name[m]+' task ('+name[n]+' label)',fontsize=20)
# plt.tight_layout()
# fig1.savefig('matplot_numfa.png')
# plt.show()


#plot the mask
data = np.load(pj(rootDir, 'mask', "TGC4D_mask_nclulster3.npy"))
import seaborn as sns
fig1 =plt.figure(figsize=(20,20),dpi=300)
plt.title('Temporal generalization mask')
count = 1
xTicks = np.arange(-98,702,5)
yTicks = np.arange(700,-100,-5)#np.arange(-100,700,2)
name = ['Num','FA']
for i in range(2):
    for j in range(2):
        for m in range(2):
            for n in range(2):
                plt.subplot(4,4,count)
                tempData = np.flip(data[i,j,m,n],axis=0) # flip upside down 
                pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
                sns.heatmap(pd_data,cmap='jet',vmin=0, vmax=1,cbar=False) #*1000/newSamplingRate,could add mask, mask =mask

                plt.ylabel('Train on '+name[i]+' task ('+name[j]+' label)',fontsize=20)
                plt.xlabel('Test on '+name[m]+' task ('+name[n]+' label)',fontsize=20)
                plt.tight_layout()
                count = count +1

fig1.savefig('matplot_numfa_mask.png')
plt.show()

'''
#plot the mask
data = np.load(pj(rootDir, "TGC4D_mask.npy"))
fig1 =plt.figure(figsize=(20,20),dpi=300)
plt.title('Temporal generalization mask')
count = 1
xTicks = np.arange(-98,702,2)
yTicks = np.arange(700,-100,-2)#np.arange(-100,700,2)
name = ['Num','FA']
for i in range(2):
    for j in range(2):
        for m in range(2):
            for n in range(2):
                plt.subplot(4,4,count)
                tempData = np.flip(data[i,j,m,n],axis=0) # flip upside down 
                pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
                sns.heatmap(pd_data,cmap='jet',vmin=0, vmax=1,cbar=False) #*1000/newSamplingRate,could add mask, mask =mask

                plt.ylabel('Train on '+name[i]+' task ('+name[j]+' label)',fontsize=20)
                plt.xlabel('Test on '+name[m]+' task ('+name[n]+' label)',fontsize=20)
                plt.tight_layout()
                count = count +1

fig1.savefig('matplot_numfa_mask.png')
plt.show()
'''
print('initing')