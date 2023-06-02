# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:44:26 2021

@author: tclem
"""
from re import I
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
# calculate partial Spearman correlation for each RDM
# ---------------------------------------------------
# '004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'
subjs = ['003'] #'002', '003'

samplingRate = 500
RDMnum = 5
# make 4 dimension x 2 (r value and p value) empty matrix
subjNums, tps= len(subjs), int(samplingRate*0.8) # 3*80

modelRDM = np.load('/data/home/nummag01/workingdir/eeg1/stimuli/ModelRDM_NumIsTfaDenLLF.npy')
RDMcorr = np.zeros((2,RDMnum,subjNums,tps))
RDMp = np.zeros((2,RDMnum,subjNums,tps))

side = 'two-sided'
partial = 'Spearman' #'partialSpearman','Spearman'
types = ['num','is'] 

print('Initing')
for j in range(len(types)):
    for subj in range(len(subjs)):
        fileName = 'ctfRDM3x100x500hz_subj' + subjs[subj] + types[j] + '.npy'
        filePath = pj(rootDir, fileName)
        # compute partial spearman correlation, with other 2 RDM controlled
        data = np.load(filePath)  # subIndex,t,re,foldIndex,RDMindex
        t, re, foldIndex, RDMindex = data.shape
        data = data.reshape(t * re * foldIndex, RDMindex)
        # normalize the MEG RDM
        #  scaler = StandardScaler()
        #  data = scaler.fit_transform(data)
        data = data.reshape(t, re, foldIndex, RDMindex)

        for tp in range(tps):
            datatmp = data[tp, :]  # subIndex,t,re,foldIndex,RDMindex
            RDMtmp = np.average(data[tp, :, :, :], axis=(0, 1))
            pdData = pd.DataFrame(
                    {'respRDM': RDMtmp, 'RDM0': modelRDM[0], 'RDM1': modelRDM[1], 'RDM2': modelRDM[2], 'RDM3': modelRDM[3],
                    'RDM4': modelRDM[4]})
            # 'numRDM','isRDM','tfaRDM','denRDM','denRDM'
            modelList = ['RDM0','RDM1', 'RDM2', 'RDM3', 'RDM4']
            if partial == 'partialSpearman':  # 'RDM0','RDM1','RDM2','RDM3','RDM4'
                for i in range(RDMnum):
                    newlist = modelList.copy()
                    del newlist[i]
                    corr = pg.partial_corr(pdData, x='respRDM', y=modelList[i],
                                        x_covar=newlist, alternative=side,
                                        method='spearman')
                    # print(str(j)+str(i)+str(subj)+str(tp))
                    RDMcorr[j, i, subj, tp] = corr['r']
                    RDMp[j, i, subj, tp] = corr['p-val']
            elif partial == 'Spearman':
                for i in range(RDMnum):
                    corr = pg.corr(RDMtmp,pdData[modelList[i]])
                    RDMcorr[j, i, subj, tp] = corr['r']
                    RDMp[j, i, subj, tp] = corr['p-val']

        del data

# plot the results
color = ["Red", "Purple", "Gray", "Blue", "Green", "Orange", 'brown']

for j in range(len(types)):
    fig = plt.figure(figsize=(9, 6), dpi=100)
    avgR = np.average(RDMcorr[j,:],axis=1)
    for i in range(RDMnum):
        plt.plot(np.arange(-100,700,1000/samplingRate),avgR[i,:],label=RDMName[i],color=color[i])
        # judge whether there are significant line
        # plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
    plt.xlabel('Time points(ms)')
    if partial == 'partialSpearman':
        plt.ylabel('Partial spearman correlation')
        # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
        plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
    elif partial =='Spearman':
        plt.ylabel('Spearman correlation')
        # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
        plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
    plt.legend()
    plt.savefig(pj(rootDir, 'GroupRSA_multifeature3'+partial+types[j]+'.png'))
    plt.show()

