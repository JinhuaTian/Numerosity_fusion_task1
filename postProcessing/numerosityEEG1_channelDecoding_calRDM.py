# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:22:51 2021

@author: tclem
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
from hashlib import new
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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
subjid = 'subj003' #subjName = ['subj016']
type = 'is' # 'num','is'
# 'subj002','subj003','subj004','subj005','subj006','subj007','subj008','subj009','subj010'

newSamplingRate = 500
repeat = 100
kfold = 3
labelNum = 45
tpoints = int(newSamplingRate*0.8) # 80*3

# compute pair number:
indexNum = 0
# calculate label pairs
labelPair = np.array([],dtype=int)
for x in range(labelNum):
    for y in range(x+1,labelNum):
        labelPair = np.hstack((labelPair,[x,y]))
        indexNum = indexNum + 1
labelPair = labelPair.reshape((-1,2))

accs = np.zeros([tpoints,repeat,kfold,indexNum]) # time,[label-1,label-1],fold,repeat

pcNum = [] # restore the number of PCs

# compute MEG RDM using pairwise SVM classification

print('subject ' + subjid +' is running.')
savePath = pj(rootDir, subjid, 'preprocessed')
epochs_list = []
# walk through subj path, concatenate single subject's data to one file
for file in os.listdir(savePath):
    if type in file:
        fifpath = pj(savePath, file)
        epoch = mne.read_epochs(fifpath, preload=False, verbose=True)
        epochs_list.append(epoch)
        del epoch
epochs_all = mne.concatenate_epochs(epochs_list)

# downsample to 500Hz
epochs_all.resample(
    sfreq=newSamplingRate,
    npad="auto",
    window="boxcar",
    # n_jobs=4,
    pad="edge",
    verbose=True)

X = epochs_all.get_data(picks = 'eeg') # exclude other sensor; # MEG signals: n_epochs, n_meg_channels, n_times
print("X shape is",str(X.shape))
# slice data -0.1~0.7s
X = X[:,:,int(newSamplingRate*0.1):tpoints+int(newSamplingRate*0.1)] # should not del it

nEpochs, nChan, nTime = X.shape
# training label
Y = epochs_all.events[:, 2]
Y = Y.reshape(nEpochs,1)
del epochs_all
# make new label
Yshape = Y.shape[0]
	
# Apply PCA to reduce dimension. It works bad, just reduce 306 dim to 276 dim
# perform pca according to the variance remains
# pcNum = []  # restore the number of PCs
X = np.transpose(X,[0,2,1]) #change (n_epochs, n_meg_channels, n_times) to (nEpochs * nTime, nChan)
X = X.reshape(nEpochs*nTime, nChan)
X = scaler.fit_transform(X) #normarlize data
pca = PCA(n_components=0.99, svd_solver="full")  # n_components=0.90,
pca = pca.fit(X)
Xpc = pca.transform(X)

# normalize PC
Xpc = scaler.fit_transform(Xpc)

# print the PC number
# pcNum.append(Xpc.shape[1])
print('PC number is '+ str(Xpc.shape[1]))
#nEpochs, nChan, nTime = X.shape
X = Xpc.reshape(nEpochs,nTime,Xpc.shape[1])
del Xpc
X = np.transpose(X,[0,2,1])

# RDM of decoding accuracy values for each time point
for t in range(nTime):
    # pick the time data and normalize data
    Xt = X[:,:,t]
    # Xt = scaler.fit_transform(Xt)

    time0 = time.time()
    #repeat for repeat times:
    for re in range(repeat):
        state = np.random.randint(0,100)
        kf=StratifiedKFold(n_splits=kfold, shuffle=True,random_state=state)
        foldIndex = 0
        for train_index, test_index in kf.split(Xt,Y):
            xTrain, xTest, yTrain, yTest, = Xt[train_index], Xt[test_index],Y[train_index],Y[test_index]
            trainPd = np.concatenate((yTrain,xTrain),axis=1) # train data
            testPd = np.concatenate((yTest,xTest),axis=1) # test data
            RDMindex = 0
            for i in range(labelPair.shape[0]): 
                x,y = labelPair[i,0],labelPair[i,1]
                Pd1 = trainPd[(trainPd[:,0] == (x+1)) | (trainPd[:,0] == (y+1))] # labels are 1~45
                Pd2 = testPd[(testPd[:,0] == (x+1)) | (testPd[:,0] == (y+1))]
                # run svm
                
                svm = SVC(kernel="linear")
                svm.fit(Pd1[:,1:],Pd1[:,0])
                acc = svm.score(Pd2[:,1:],Pd2[:,0]) # sub,time,RDMindex,fold,repeat # t,re,foldIndex,RDMindex
                # save acc
                accs[t, re, foldIndex, RDMindex] = acc
                '''
                model = sv.svm_train(Pd1[:,0],Pd1[:,1:], "-q -s 0 -t 0")
                # see for details https://github.com/cjlin1/libsvm/blob/master/python/README 
                _, p_acc, _ = sv.svm_predict(Pd2[:,0],Pd2[:,1:], model,"-q")# p_acc: a tuple including accuracy (for classification), mean squared error, and squared correlation coefficient (for regression).
                # save acc
                accs[t,re,foldIndex,RDMindex]=p_acc[0]
                '''
                RDMindex = RDMindex + 1
            foldIndex = foldIndex + 1
    time_elapsed = time.time() - time0
    print('Time point {} finished in {:.0f}m {:.0f}s'.format(t, time_elapsed // 60, time_elapsed % 60)) # + 'repeat '+ str(re)


# save MEG RDM
fileName = pj(rootDir, 'ctfRDM3x100x500hz_'+ subjid+type +'.npy')
np.save(fileName,accs)

'''
partialAvgAcc = np.average(accs, axis=(2, 3, 4))

import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-30,x[0]-30),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')
'''
