# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:44:26 2021

@author: tclem
"""
from cProfile import label
from re import I
from hypothesis import event
import numpy as np
# import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import join as pj
# from neurora.stuff import clusterbased_permutation_1d_1samp_1sided as clusterP
import scipy.stats as st
import matplotlib.pyplot as plt
# load stimulus RDM
rootDir = '/data/home/nummag01/workingdir/eeg1/'
# make correlation matrix
RDMName = ['Number', 'Item area', 'Total field area', 'Density', 'Low-level Feature']
# ---------------------------------------------------
#
# ---------------------------------------------------
# '004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'
subjs = ['002','003'] #'002', '003'

samplingRate = 500
RDMnum = 5
labelNum = 45
# make 4 dimension x 2 (r value and p value) empty matrix
subjNums, tps= len(subjs), int(samplingRate*0.8) # -0.1~0.7

modelRDM = np.load('/data/home/nummag01/workingdir/eeg1/stimuli/ModelRDM_NumIsTfaDenLLF.npy')
RDMcorr = np.zeros((2,RDMnum,subjNums,tps))
RDMp = np.zeros((2,RDMnum,subjNums,tps))
eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/STI2.txt')

side = 'two-sided'
partial = 'Spearman' #'partialSpearman','Spearman'
types = ['num','is'] 
# reconstruct RMD using image pair
def reconstrRDM(RDM): # averaged RDM dimensions: tps x label pairs
    rdmMatrix = np.zeros((tps,labelNum,labelNum))
    count = 0
    for x in range(labelNum):
        for y in range(x+1,labelNum):
            for tp in range(tps):
                rdmMatrix[tp,x,y] = RDM[tp,count]
                rdmMatrix[tp,y,x] = RDM[tp,count]
            count = count + 1
    return rdmMatrix
print('Initing')
# load data
fullData = np.zeros((len(types),len(subjs),tps,labelNum,labelNum))
# accs = np.zeros([tpoints,repeat,kfold,indexNum])
for j in range(len(types)):
    for subj in range(len(subjs)):
        fileName = 'ctfRDM3x100x500hz_subj' + subjs[subj] + types[j] + '.npy'
        filePath = pj(rootDir, fileName)
        # compute partial spearman correlation, with other 2 RDM controlled
        data = np.load(filePath)  # tpoints,repeat,kfold,indexNum
        data = np.average(data,axis=(1,2))
        fullData[j,subj,:,:,:] = reconstrRDM(data)
        del data

# slice time, apply MDS, and plot the results
Name = ['Number','Item Area']

from sklearn.manifold import MDS,TSNE
import matplotlib.pyplot as plt
dpi = 300
# randState = 2
tsne = TSNE(n_components=2, learning_rate=200, n_iter=1000, init='pca', random_state=500)
# random_state=randState,
mds = MDS(n_components=2, dissimilarity='precomputed',metric=True) # If True, perform metric MDS; otherwise, perform nonmetric MDS.
method = 'MDS'
timeWindow = [108,138,143,185] # 0.116，0.176，0.186，0.27 + 0.1sf f
peakName = ['N1','P1','N2','P3']
eventName = ['Num1IA1','Num1IA2','Num1IA3','Num2IA1','Num2IA2','Num2IA3','Num3IA1','Num3IA2','Num3IA3']
color = ['r','r','r','b','b','b','black','black','black']#['b','b','b','orange','orange','orange','green','green','green']
size = [6,12,18,6,12,18,6,12,18]
for j in range(len(types)):
    avgData = np.average(fullData[j,:,:,:],axis=0)
    for i in range(len(timeWindow)): #-0.2~0.8s
        summary = mds.fit_transform(avgData[timeWindow[i],:,:])
        plt.figure(figsize=(9, 6), dpi=dpi)
        for m in range(len(eventName)):
            loc = eventMatrix[:,3]==m+1
            plt.scatter(summary[loc,:][:,0],summary[loc,:][:,1],c=color[m],s=size[m],label=eventName[m])
        plt.title('MDS of '+Name[j]+' on time point '+str(timeWindow[i]*2-100)+' ms '+'('+peakName[i]+')')
        plt.legend(loc='upper right')
        plt.savefig(Name[j]+peakName[i]+'.png')
        # plt.show()    

print("Done")