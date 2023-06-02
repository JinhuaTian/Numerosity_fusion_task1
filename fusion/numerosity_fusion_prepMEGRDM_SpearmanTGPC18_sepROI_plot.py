# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:22:51 2022

@author: tclem
"""
'''
2 tasks x tx x ty x RDMIndex
'''
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
# import matplotlib
# matplotlib.use('Qt5Agg') #TkAgg
#from libsvm import svmutil as sv # pip install -U libsvm-official
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/fMRI'
saveDir = '/data/home/nummag01/workingdir/fusion1/fMRI/sepROI'
# nohup python numerosity_channelDecoding_RSAcalRDM.py > rsa2.out 2>1&1 & 
subjs = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025']

types = ['num','fa'] # 'num','fa'
# basic info
newSamplingRate = 500
labelNum = 45
levelNum = 9
tpoints = int(newSamplingRate*0.8) # 80*3

ROInames = ['V3d(L)','V3d(R)','IPS(L)','SFG(L)'] 
# roiList = [[1,2,3,4,5,6,16,17],[8,9],[10,11],[14,15],[18,19,20,21,22,23,24],[25]]
imgNum=9 
taskNum =2 
taskTypes = ['Judge Number','Judge Field Area']

numRDV = np.load(pj(saveDir,'numRDV_subjRoiVec.npy')) # nsubject x nROI x nRDV
faRDV = np.load(pj(saveDir,'faRDV_subjRoiVec.npy')) # nsubject x nROI x nRDV
fMRIrdv = [numRDV,faRDV] # !!![ntask][nsubject x nROI x nRDV]

fileName = pj(rootDir, 'ctfRDM_TGPC18x'+str(newSamplingRate)+'hz.npy')
MEGrdv = np.load(fileName) # subjeces x task x time points x time points x RDV length

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# plot RDM for each task, each ROI
corrMatrix = np.zeros((len(subjs),len(taskTypes),len(ROInames),tpoints,tpoints))
for subjNum in range(len(subjs)):
    for taskN in range(len(taskTypes)):
        for roi in range(len(ROInames)):
            for tx in range(tpoints):
                for ty in range(tpoints):
                    corrMatrix[subjNum,taskN,roi,tx,ty],_ = spearmanr(MEGrdv[subjNum,taskN,tx,ty,:],fMRIrdv[taskN][subjNum,roi,:])
# average cross subjects
avgCorrMatrix = np.average(corrMatrix,axis=0)
# plot temporal generalization matrix for each ROI
for taskNum in range(len(taskTypes)):
    count = 1
    fig1 =plt.figure(figsize=(20,20),dpi=300)
    for roi in range(len(ROInames)):        
        #plt.title('Temporal generalization')        
        xTicks = np.arange(-98,702,5)
        yTicks = np.arange(700,-100,-5)#np.arange(-100,700,2)
        locator = np.arange(0,800,100)

        ax = plt.subplot(2,2,count) # four ROIs
        tempData = np.flip(avgCorrMatrix[taskNum,roi,:,:],axis=0) # flip upside down 
        pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
        sns.heatmap(pd_data,cmap='jet',cbar=True, vmin=0, vmax=0.2) # ,vmin=0.2, vmax=0.5*1000/newSamplingRate,could add mask, mask =mask
        plt.title('MEG-'+ROInames[roi],fontsize=30)      
        plt.ylabel('Time (ms)',fontsize=20)
        plt.xlabel('Time (ms)',fontsize=20)
        
        count = count +1
    plt.tight_layout()
    fig1.savefig('matplot_numfa'+taskTypes[taskNum]+'.png')
    plt.show()
# plot RSA-based TG
rawMEGTG = np.average(MEGrdv,axis=(0,1,4))
fig1 =plt.figure(figsize=(20,20),dpi=300)    
#plt.title('Temporal generalization')        
xTicks = np.arange(-98,702,5)
yTicks = np.arange(700,-100,-5)#np.arange(-100,700,2)
locator = np.arange(0,800,100)
ax = plt.subplot(1,1,1) # four ROIs
tempData = np.flip(rawMEGTG,axis=0) # flip upside down 
pd_data=pd.DataFrame(tempData,index=yTicks,columns=xTicks)
sns.heatmap(pd_data,cmap='jet',cbar=True) # ,vmin=0.2, vmax=0.5*1000/newSamplingRate,could add mask, mask =mask
#plt.gca().xaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600])) 
#plt.gca().yaxis.set_major_locator(ticker.FixedLocator([0,100,200,300,400,500,600]))     
plt.title('MEG TG',fontsize=30)      
plt.ylabel('Time (ms)',fontsize=20)
plt.xlabel('Time (ms)',fontsize=20)
count = count +1
plt.tight_layout()
fig1.savefig('matplot_MEG.png')
plt.show()

print('All DONE')
print('All DONE')