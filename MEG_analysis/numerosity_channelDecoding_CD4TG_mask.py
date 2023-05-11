#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:09:15 2021

@author: tianjinhua

calculate the TG and its mask
"""
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
scaler = StandardScaler()
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4'

subjList = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025'] #,'subj011','subj012','subj013','subj017','subj018'
eventMatrix = np.loadtxt(pj(rootDir,'STI2.txt'),dtype=int)

titles = ['number','field area']
newSamplingRate = 200
labelNum = 80
tpoints = int(newSamplingRate*0.8) # -100 ~ 700ms
dims = 2
subjNum = len(subjList)
accs = np.zeros((subjNum,dims,dims,dims,dims,tpoints,tpoints))
i = 0
for subj in subjList:
    subjDir = pj(rootDir,'crossDecoding_CD22TG_svm12x2x2x'+str(newSamplingRate)+'hz_'+ subj +'.npy')
    subjData = np.load(subjDir)
    accs[i] = subjData 
    i = i+1

'''
def plotMatrix(data, title, vmin=0.45, vmax=0.55):
    data = pd.DataFrame(data,dtype='float64')
    fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
    cax = fig.add_subplot(111)
    cax = cax.matshow(data, vmin=vmin, vmax=vmax)  # cmap='jet',绘制热力图，从-1到1  ,
    ax = plt.gca()
    #ax.invert_xaxis()
    ax.invert_yaxis()
    fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
    #ax.set_xticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
    #ax.set_yticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
    # plt.xlim(np.linspace(-100, 100, 700))
    plt.ylabel('Training time')
    plt.xlabel('Testing time')
    plt.title(title)
    #plt.show()
'''
'''
count = 0
vmin,vmax=0.48,0.52
fig1 = plt.figure(dpi=300)
#accs = accs/newSamplingRate*1000
avgaccs = np.average(accs,axis=0)
for dec in range(dims):
    for dim in range(dims):
        ax1 = fig1.add_subplot(dims,dims,count+1)        
        data = pd.DataFrame(avgaccs[dec,dim,:,:],dtype='float64')
        # fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
        #cax = fig.add_subplot(111)
        ax1 = ax1.matshow(data, vmin=vmin, vmax=vmax)  # cmap='jet',绘制热力图，从-1到1  ,
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=4)
        ax.tick_params(axis='y', labelsize=4)
        #ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('bottom')
        # fig.colorbar(ax1)  #cax将matshow生成热力图设置为颜色渐变条
        #ax.set_xticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
        #ax.set_yticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
        # plt.xlim(np.linspace(-100, 100, 700))
        #plt.ylabel('Training time')
        #plt.xlabel('Testing time')
        # plt.title(title)
        
        # ax1.imshow(fig)
        count = count+1
fig1.colorbar(ax1)
plt.tight_layout()
plt.show()
'''

from jhTools import clusterbased_permutation_1d_1samp_1sided # , permutation_diff
from neurora.stuff import clusterbased_permutation_2d_1samp_1sided as cluster2d

dimPairs = []
for dim1 in range(dims):
    for dim2 in range(dims):
        for dim3 in range(dims):
            for dim4 in range(dims):
                dimPairs.append([dim1,dim2,dim3,dim4])
from joblib import Parallel, delayed
n_threshold=4 # 2d cluster
def calMask(i):
    mask = cluster2d(accs[:,dimPairs[i][0],dimPairs[i][1],dimPairs[i][2],dimPairs[i][3],:,:],level=0.3334,n_threshold=n_threshold)
    np.save(pj(rootDir,'mask', 'TGC4D_mask'+str(i)+'.npy'), mask)

# 18 subjects in total4
Parallel(n_jobs=16)(delayed(calMask)(i) for i in range(16))
'''
n_threshold=3
def calMask(i):
    mask = np.zeros((tpoints,tpoints)) #16 masks for each dimensions 
    for tp in range(tpoints):
        mask[tp] = clusterbased_permutation_1d_1samp_1sided(accs[:,dimPairs[i][0],dimPairs[i][1],dimPairs[i][2],dimPairs[i][3],tp,:],level=0.3334,n_threshold=n_threshold)
#    np.save(pj(rootDir,'mask', 'TGC4D_mask'+str(i)+'.npy'), mask)

# 18 subjects in total
Parallel(n_jobs=16)(delayed(calMask)(i) for i in range(16))
'''

# load all mask data and save the mask
data = np.zeros((dims,dims,dims,dims,tpoints,tpoints)) # mask data
for i in range(16):
    mask = np.load(pj(rootDir, 'mask', 'TGC4D_mask'+str(i)+'.npy'))
    data[dimPairs[i][0],dimPairs[i][1],dimPairs[i][2],dimPairs[i][3],:,:] = mask
# save concated masks
np.save(pj(rootDir,'mask', 'TGC4D_mask_nclulster'+str(n_threshold)+'.npy'), data)

#plot the mask
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

print("All Done")