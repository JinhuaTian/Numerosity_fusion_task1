#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:13:38 2021

@author: tianjinhua
"""

'''
CalRDM:
1. concatenate all data at subject level
2. down-sample data to 300 hz, slice data -0.1~0.7s
3. arrange data: picks=meg, normalize the data(mne.decoding.Scaler)
4. compute MEG RDM: pairwise SVM classification
Run RSA:
5. compute model RDM: number, field size, item size
    arrange label
6. compute the partial Spearman correlation between MEG RDM and model RDM
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

from functools import reduce
from operator import add

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
# 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
# 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025'
subjList = ['subj016'] 
#eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/stimuli/ModelRDM_NumIsTfaDenLLF.npy',encoding='unicode_escape',dtype=int)
eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/fusion1/MEG4/STI2.txt')
newSamplingRate = 200
labelNum = 45
tpoints = int(newSamplingRate*0.8) #-0.1~0.7ms
session = 14 # or 11
decodingNum = 2 #'num', 'fa'
def applyPCA(X,nComp=0.9999):
    # Apply PCA to reduce dimension. It works well
    # perform pca according to the variance remains
    nEpochs, nChan, nTime = X.shape
    X = np.transpose(X,[0,2,1]) #change (n_epochs, n_meg_channels, n_times) to (nEpochs * nTime, nChan)
    X = X.reshape(nEpochs*nTime, nChan)
    pca = PCA(n_components=nComp, svd_solver="full")  # n_components=0.90,
    pca = pca.fit(X)
    Xpc = pca.transform(X)
    # print the PC number
    print('PC number is '+ str(Xpc.shape[1]))
    #nEpochs, nChan, nTime = X.shape
    X = Xpc.reshape(nEpochs,nTime,Xpc.shape[1])
    del Xpc
    X = np.transpose(X,[0,2,1])
    return X

def transLabel(label):
    label = np.array(label)
    label0,label1 = np.zeros(label.shape[0],dtype=int),np.zeros(label.shape[0],dtype=int)
    for i in range(labelNum):
        label0[label == i] = eventMatrix[:,0][eventMatrix[:,2]==i]
        label1[label == i] = eventMatrix[:,1][eventMatrix[:,2]==i]
    return label0,label1

def flattenList(listlist):
    resultList = []
    for childList in listlist:
        resultList = resultList+childList.tolist() # transfer numpy to list
    return resultList

trialNum = 90 # each number or fa run

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
    for sess in range(1,session+1):
        fifpath1 = pj(savePath, 'num'+str(sess)+'.fif')
        epoch1 = mne.read_epochs(fifpath1, preload=True, verbose=True)
        # epoch1.info['dev_head_t'] = Transform('meg', 'head', np.identity(4)) # ctf use mag instead of meg
        fifpath2 = pj(savePath, 'fa'+str(sess)+'.fif')
        epoch2 = mne.read_epochs(fifpath2, preload=True, verbose=True)
        # epoch2.info['dev_head_t'] = Transform('meg', 'head', np.identity(4)) # ctf use mag instead of meg

        # extract label
        # if remaining epochs of a block >50%, keep this block
        if epoch1.events[:, 2].shape[0] > (trialNum*0.5) and epoch2.events[:, 2].shape[0] > (trialNum*0.5):
            # select label array
            # modify Y 1~50 ->1~25
            Y1 = epoch1.events[:, 2] #np.ceil(epoch1.events[:, 2]/2).astype(int)
            labelList1.append(Y1)        
            epochData1 = epoch1.get_data(picks = 'mag') # 273 instead of all 301 channels
            print(epochData1.shape)

            # modify Y 1~50 ->1~25
            Y2 = epoch2.events[:, 2] # np.ceil(epoch2.events[:, 2]/2).astype(int)
            labelList2.append(Y2)        
            epochData2 = epoch2.get_data(picks = 'mag')
            
            # select label, epoch number  
            nEpochs1, _, _ = epochData1.shape
            nEpochs2, _, _ = epochData2.shape

            epochList1.append(list(range(epochCount1,epochCount1+nEpochs1)))
            epochList2.append(list(range(epochCount2,epochCount2+nEpochs2)))
            
            epochCount1 = epochCount1 + nEpochs1
            epochCount2 = epochCount2 + nEpochs2
            
            epochs_list1.append(epoch1)
            epochs_list2.append(epoch2)
            del epoch1, epochData1,epoch2, epochData2
    
    # count the remaining epoch number
    sessNum = len(epochs_list1) 
    accs = np.zeros([sessNum,decodingNum,decodingNum,decodingNum,decodingNum,tpoints,tpoints]) # sessions(12) x train data x label x test data x label x tpoints
    epochs_all1 = mne.concatenate_epochs(epochs_list1)
    epochs_all2 = mne.concatenate_epochs(epochs_list2)
    
    # downsample to 500Hz
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
    X1 = epochs_all1.get_data(picks = 'mag') # exclude other sensor; # MEG signals: n_epochs, n_meg_channels, n_times
    X2 = epochs_all2.get_data(picks = 'mag')
    print('Data1 shape is '+ str(X1.shape))
    print('Data2 shape is '+ str(X2.shape))
    # slice data -0.1~0.7s
    X1 = X1[:,:,int(newSamplingRate*0.1):tpoints+int(newSamplingRate*0.1)] # should not del it
    X2 = X2[:,:,int(newSamplingRate*0.1):tpoints+int(newSamplingRate*0.1)]

    # PCA, compare two data components
    X1test = applyPCA(X1)
    X2test = applyPCA(X2)

    if X1test.shape[1] != X2test.shape[1]:
        minPC = min([X1test.shape[1],X2test.shape[1]])
        X1 = applyPCA(X1,nComp=minPC)
        X2 = applyPCA(X2,nComp=minPC)
    elif X1test.shape[1] == X2test.shape[1]:
        X1 = X1test
        X2 = X2test
    del X1test,X2test

    nEpochs1 = X1.shape[0]
    nEpochs2 = X2.shape[0]

    # # training label
    # Y1 = epochs_all1.events[:, 2]
    # Y1 = Y1.reshape(nEpochs1,1)
    
    # Y2 = epochs_all2.events[:, 2]
    # Y2 = Y2.reshape(nEpochs2,1)
    
    del epochs_all1,epochs_all2
    # make new label
    # Yshape = Y.shape[0]
    # print("X shape is",str(X.shape))
    
    for sess in range(sessNum):
        time0 = time.time()
        # segement the train test data: exclude test data and then concatenate data
        tempTrain1 = epochList1.copy()
        del tempTrain1[sess]
        tempTrain1 = reduce(add,tempTrain1) # flattern the list
        tempTrain1 = X1[tempTrain1,:,:]

        tempTrain2 = epochList2.copy()
        del tempTrain2[sess]
        tempTrain2 = reduce(add,tempTrain2)
        tempTrain2 = X2[tempTrain2,:,:]
        trainData = [tempTrain1,tempTrain2]
        # 2 train label: data format: list(np.array...)
        tempLabel1 = labelList1.copy()
        del tempLabel1[sess]
        tempLabel1 = flattenList(tempLabel1)
        label10,label11 = transLabel(tempLabel1)
        label1 = [label10,label11]

        tempLabel2 = labelList2.copy()
        del tempLabel2[sess]
        tempLabel2 = flattenList(tempLabel2)
        label20,label21 = transLabel(tempLabel2)
        label2 = [label20,label21]

        trainLabel = [label1,label2]
        # 2 test data
        testData1 = X1[epochList1[sess],:,:]
        testData2 = X2[epochList2[sess],:,:]
        testData = [testData1,testData2]
        # 2 test label
        testLabel1 = np.array(labelList1[sess])
        testLabel10,testLabel11 = transLabel(testLabel1)
        testLabel1 = [testLabel10,testLabel11]

        testLabel2 = np.array(labelList2[sess])
        testLabel20,testLabel21 = transLabel(testLabel2)
        testLabel2 = [testLabel20,testLabel21]

        testLabel = [testLabel1,testLabel2]
        for tp1 in range(tpoints):
            for trainDat in range(2): # 2 training datasets
                for trainLab in range(2): # 2 labels for each data sets
                    #model = LinearDiscriminantAnalysis()
                    model = SVC(kernel="linear")
                    model.fit(trainData[trainDat][:,:,tp1],trainLabel[trainDat][trainLab])
                    for tp2 in range(tpoints): # test on other datasets, labels, and time points
                        for testDat in range(2): # 2 testing datasets
                            for testLab in range(2): # 2 testing label
                                acc = model.score(testData[testDat][:,:,tp2],testLabel[testDat][testLab]) # sub,time,RDMindex,fold,repeat # t,re,foldIndex,RDMindex
                                # save acc
                                accs[sess,trainDat,trainLab,testDat,testLab,tp1,tp2] = acc   
        time_elapsed = time.time() - time0
        del trainData,trainLabel,testData,testLabel
        print('session number {} finished in {:.0f}m {:.0f}s'.format(sess, time_elapsed // 60, time_elapsed % 60)) # + 'repeat '+ str(re)        
    del X1,X2,Y1,Y2
    accs = np.average(accs,axis=0)
    # save MEG RDM
    fileName = pj(rootDir, 'crossDecoding_CD22TG_svm12x2x2x'+str(newSamplingRate)+'hz_'+ subjid +'.npy')
    np.save(fileName,accs)
    print('subject finished')

print('All Done')