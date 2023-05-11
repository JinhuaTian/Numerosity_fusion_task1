# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:31:05 2023

@author: tclem
"""
from os.path import join as pj
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#import torchvision.transforms as transforms
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

path = r'D:\坚果云\我的坚果云\毕业论文\meg2\modelRDM.npy'
RDMs = np.load(path)

def flipDiag(matrix):
    matrix = np.flip(matrix,axis=0)
    matrix = np.flip(matrix,axis=1)
    return matrix

plotPics = ['数','占据视野','密度','像素相关']
import seaborn as sns
for i in range(len(plotPics)):
    fig = plt.figure(figsize=(8, 6), dpi=100) #è°ç¨figureåå»ºä¸ä¸ªç»å¾å¯¹è±¡
    ax = fig.add_subplot(111)

    RDMs[i] = flipDiag(RDMs[i])
    #cax = ax.matshow(RDMs[i], cmap='jet',vmin=0, vmax=1)  #ç»å¶ç­åå¾ï¼ä»-1å°1  ,
    #fig.colorbar(cax)  #caxå°matshowçæç­åå¾è®¾ç½®ä¸ºé¢è²æ¸åæ¡
    mask = np.triu(np.ones_like(RDMs[i],dtype=bool))
    sns.heatmap(RDMs[i], mask=mask, cmap='jet',vmin=0,vmax=1) #, robust=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}
    #plt.title(plotPics[i] +'RDM',fontsize=20)
    plt.savefig('model'+str(i))
    plt.show()