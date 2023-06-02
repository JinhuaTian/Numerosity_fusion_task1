# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:57:45 2021

@author: tclem

calculate Number, field size, item size, shape, density (field size/number), TFA (item size x number) 
"""
from os.path import join as pj
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#import torchvision.transforms as transforms
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

eventMatrix =  np.loadtxt('/data/home/nummag01/workingdir/eeg1/STI2.txt')

# make correlation matrix
index = 0
numRDM = []
isRDM = []
denRDM = []
tfaRDM = []
#LLFRDM = np.load('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy') # low-level features

labelNum = 45
# compute model RDM
for x in range(labelNum):
    for y in range(x+1,labelNum):
        numRDM.append(abs(eventMatrix[x,0]-eventMatrix[y,0]))
        isRDM.append(abs(eventMatrix[x,1]-eventMatrix[y,1]))
        denRDM.append(abs(eventMatrix[x,1]/eventMatrix[x,0]-eventMatrix[y,1]/eventMatrix[y,0]))
        tfaRDM.append(abs(eventMatrix[x,1]*eventMatrix[x,0]-eventMatrix[y,1]*eventMatrix[y,0]))
            
# calculate low-level feature
stiPath = '/data/home/nummag01/workingdir/eeg1/stimuli'
folders = ['Num6_IA1','Num6_IA2','Num6_IA3','Num10_IA1','Num10_IA2','Num10_IA3','Num17_IA1','Num17_IA2','Num17_IA3']
imgRepeat = 5
imgs = []

# make empty image matrix
img = Image.open(pj(stiPath,folders[0],'1.png'))
img = img.convert('1') # should not use "L"
# img.show()
img = np.array(img)

a,b=img.shape
imgNum = 45  

imgMatrix = np.zeros((imgNum,a*b))

num = 0
for f in folders:
    for i in range(imgRepeat):
        imgpath = pj(stiPath,folders[0],str(i+1)+'.png')
        imgfile = Image.open(imgpath)
        # convert to gray scale
        imgfile = imgfile.convert('L')
        imgfile = np.array(imgfile)
        imgfile = imgfile.reshape(a*b)
        imgMatrix[num,:]=imgfile
        num = num + 1

'''
# normarlize image data
scaler = StandardScaler()
imgMatrix = scaler.fit_transform(imgMatrix)
'''

LLFRDM = np.zeros((imgNum,imgNum))
corrArray = []
TFAArray = []

# RDM, use 1-r instead
for x in range(labelNum):
    for y in range(x+1,labelNum):
        r = pearsonr(imgMatrix[x,:],imgMatrix[y,:])[0]
        corrArray.append(1-r)
        LLFRDM[x,y] = 1-r

#nomarlize to [0,1]
def normalization(data):
    data = np.array(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
'''
def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr
'''

# normalize RDM
# LLFRDM = normalization(LLFRDM) # no need to normalize data
# nomalize RDM vector
numRDM = normalization(numRDM)
isRDM = normalization(isRDM)
denRDM = normalization(denRDM)
tfaRDM = normalization(tfaRDM)
corrArray = normalization(corrArray) # LLF RDM vector, half of the RDM


# plot LLF RDM
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6), dpi=300) #??figure????????
ax = fig.add_subplot(111)
cax = ax.matshow(LLFRDM, cmap='jet',vmin=0, vmax=1)  #???????-1?1  ,
fig.colorbar(cax)  #cax?matshow?????????????
plt.title('Low-level feature RDM')
plt.show()
'''
# plot TFA RDM
fig = plt.figure(figsize=(6, 6), dpi=300) #??figure????????
ax = fig.add_subplot(111)
cax = ax.matshow(TFARDM, cmap='jet',vmin=0, vmax=1)  #???????-1?1  ,
fig.colorbar(cax)  #cax?matshow?????????????
plt.title('Low-level feature RDM')
plt.show()
'''
def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

RDMs = [numRDM,isRDM,tfaRDM,denRDM,corrArray]
plotPics = ['Number','Item area', 'Total Field area', 'Density','Low-level feature']
'''
for i in range(len(plotPics)):
    fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(RDMs[i], cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
    fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
    plt.title(plotPics[i] +' RDM')
    plt.show()
'''

from pandas.core.frame import DataFrame
import pandas as pd
corrMatrix = DataFrame([numRDM,isRDM,tfaRDM,denRDM,corrArray])            
corr_all = corrMatrix.T.corr(method='spearman')

fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(corr_all, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
ax.set_xticklabels(['','Number','Item area', 'Total Field area', 'Density','Low-level feature'],fontdict={'size': 10, 'color': 'black'})
ax.set_yticklabels(['','Number','Item area', 'Total Field area', 'Density','Low-level feature'],fontdict={'size': 10, 'color': 'black'})
plt.title('Spearman correlation five RDM')
plt.show()

modelRDM = np.array([numRDM,isRDM,tfaRDM,denRDM,corrArray])
path = pj(stiPath,'ModelRDM_NumIsTfaDenLLF.npy')
np.save(path,modelRDM) 