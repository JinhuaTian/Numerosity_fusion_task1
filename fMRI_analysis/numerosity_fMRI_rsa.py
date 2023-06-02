import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import nibabel
# fMRI=[15,5,20,7,12,11,16,4,3,18,17,14,13,10,2,8,19,6]
MEG = [1,2,3,4,5,6,7,8,11,12,16,17,18,19,21,23,24,25]
MEGDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
roiPath = '/data/home/nummag01/workingdir/fusion1/funcROI/'
saveDir = '/data/home/nummag01/workingdir/fusion1/fMRI/sepROI'
# load contrast 
#ROIname = ['Early visual cortex(EVC)', '(VO)','lateral Occipical Cortex(LO)',,'Frontal eye field(FEF)']
#ROInameShort = ['EVC','VO','PHC','LO','IPS','FEF'] # 'PSL'?
ROIname = ['V3d(L)','V3d(R)','IPS(L)','SFG(L)'] 
# roiList = [[1,2,3,4,5,6,16,17],[8,9],[10,11],[14,15],[18,19,20,21,22,23,24],[25]]
imgNum=9 
taskNum =2 
taskType = ['Judge Number','Judge Field Area']
method = ['direct','com']

newSamplingRate = 500
tpoints = int(newSamplingRate*0.8) # 80*3
# load RDV (OR RSV)
numRDV = np.load(pj(saveDir,'numRDV_subjRoiVec.npy')) # nsubject x nROI x nRDV
faRDV = np.load(pj(saveDir,'faRDV_subjRoiVec.npy')) # nsubject x nROI x nRDV
fMRIrdv = [numRDV,faRDV] # !!![ntask][nsubject x nROI x nRDV]
del numRDV, faRDV
# load model RDV (the first two models)
modelRDM = np.load(pj(MEGDir,'ModelRDM_NumFaDenLLF9.npy'))

RDVlength = 0
for x in range(imgNum):
    for y in range(x+1,imgNum):
        RDVlength = RDVlength+1
#  2 tasks, subjects,roi
RDMName = ['Number', 'Field area', 'Density']# , 'Low-level Feature']
modelList = ['RDM0','RDM1', 'RDM2'] #, 'RDM3']
corrResults = np.zeros((len(taskType),len(MEG),len(ROIname),len(modelList))) 

#computing Spearman's rank-order correlations between fMRI dissimilarity matrices
from scipy.stats import spearmanr
from pingouin import correlation as pg
import matplotlib.pyplot as plt
import pandas as pd
side = 'two-sided'
partial = 'partialSpearman' #'partialSpearman','Spearman','kendall'
dpi = 100
for taskN in range(len(taskType)):
    for subjN in range(len(MEG)):
        for roiN in range(len(ROIname)):
            fMRIVector = fMRIrdv[taskN][subjN,roiN,:]
            pdData = pd.DataFrame(
                {'fMRI': fMRIVector, 'RDM0': modelRDM[0], 'RDM1': modelRDM[1], 'RDM2': modelRDM[2]})  #, 'RDM3': modelRDM[3]
            # three common apporch for RSA
            if partial == 'partialSpearman':
                for i in range(len(modelList)):
                    newlist = modelList.copy()
                    del newlist[i]
                    corr = pg.partial_corr(pdData, x='fMRI', y=modelList[i],
                                        x_covar=newlist, alternative=side,
                                        method='spearman')
                    # print(str(j)+str(i)+str(subj)+str(tp))
                    corrResults[taskN,subjN,roiN,i] = corr['r']
            elif partial == 'Spearman':
                for i in range(len(modelList)):
                    corr = pg.corr(fMRIVector,pdData[modelList[i]])
                    corrResults[taskN,subjN,roiN,i] = corr['r']
            elif partial == 'kendall':
                for i in range(len(modelList)):
                    corr = pg.corr(fMRIVector,pdData[modelList[i]],alternative='two-sided',method='kendall')
                    corrResults[taskN,subjN,roiN,i] = corr['r']
# plot RSA results for each ROI 
import seaborn as sns
import matplotlib.pyplot as plt
for taskN in range(len(taskType)):
    for roiN in range(len(ROIname)):
        fig1 = plt.figure(figsize=(9,6),dpi=100)
        fullDict = {}
        for modelCount in range(len(RDMName)):
            fullDict[RDMName[modelCount]] = corrResults[taskN,:,roiN,modelCount]
        # {modelName[0]: params[:,0], modelName[1]: params[:,1], modelName[2]: params[:,2], modelName[3]: params[:,3],modelName[4]: params[:,4],modelName[5]: params[:,5],modelName[6]: params[:,6]}
        dataMax = pd.DataFrame(fullDict) #, 'Low-level featrure': maxR[m,4]
        #dataMax = rearrange(dataMax,ascending=False)    
        #sns.violinplot(data=dataMax)
        sns.barplot(data=dataMax) #,errorbar="sd"
        #plt.ylim(0,0.6)
        #plt.title('Correlation coefficient difference task)')
        plt.title('Task: '+taskType[taskN]+', ROI: '+ROIname[roiN])
        plt.ylabel('Beta weights')
        plt.legend()
        plt.tight_layout()
        plt.savefig(pj(saveDir, 'pic/rsa'+partial+taskType[taskN]+ROIname[roiN]+'.png'))
        plt.show()

# plot RSA results
# 2 picutres: modelType, subject X ROI name 
# corrResults = np.zeros((len(taskType),len(MEG),len(ROIname),len(modelList))) 
import pandas as pd
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
corrResults = corrResults.transpose((0,2,1,3)) # modelList,
for taskN in range(len(taskType)):
    plotData=corrResults[taskN,:,:,:].copy()
    plotData=plotData.reshape((len(ROIname),len(MEG)*len(modelList)))
    modelType = []
    for j in range(len(MEG)):
        modelType=modelType+RDMName

    newPd = pd.DataFrame({'modelName': modelType})
    for i in range(len(ROIname)):
        newPd[ROIname[i]] = plotData[i,:]
    std_table = newPd.groupby(by = 'modelName').sem() #.std() # ?????
    figdata = newPd.groupby(by = 'modelName').mean() #????
    plt.errorbar(figdata.columns,figdata.loc[RDMName[0]],yerr=std_table.loc[RDMName[0]],fmt='k-o',lw = 2,color='r',ecolor='r',elinewidth=1,ms=7,capsize=3,label=RDMName[0]) 
    plt.errorbar(figdata.columns,figdata.loc[RDMName[1]],yerr=std_table.loc[RDMName[1]],fmt='k-x',lw = 2,color='b',ecolor='b',elinewidth=1,ms=7,capsize=3,label=RDMName[1])
    plt.errorbar(figdata.columns,figdata.loc[RDMName[2]],yerr=std_table.loc[RDMName[2]],fmt='k-d',lw = 2,color='orange',ecolor='orange',elinewidth=1,ms=7,capsize=3,label=RDMName[2])
    #plt.errorbar(figdata.columns,figdata.loc[RDMName[3]],yerr=std_table.loc[RDMName[3]],fmt='k-s',lw = 2,ecolor='green',elinewidth=1,ms=7,capsize=3,label=RDMName[3])
    
    # plt.errorbar(figdata.columns,figdata.loc['R2D2'],yerr=std_table.loc['R2D2'],fmt='k-h',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
    # plt.errorbar(figdata.columns,figdata.loc['R2D3'],yerr=std_table.loc['R2D3'],fmt='k-^',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
    #Songti = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc') #
    plt.xlabel('ROI name', fontsize=14) # fontproperties=Songti,
    plt.ylabel('Beta weights',fontsize=14)
    plt.title('Task: '+taskType[taskN], fontsize=16)
    #myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STKAITI.ttf',
    #                        size=10)
    #plt.legend(prop=myfont, fontsize=19,ncol=2)
    #plt.savefig(r'C:\Users\Administrator\Desktop\{}.jpg'.format('Upper light interception'), dpi=400)
    #plt.savefig(r'C:\Users\13290\Desktop\{}light interception.svg'.format(name[i]), format='svg') #  ?????
    plt.ylim(-0.2,0.6)
    plt.tight_layout()
    plt.legend()
    plt.savefig(pj(saveDir, 'pic/rsa'+partial+taskType[taskN]+'.png'))
    plt.show()
from scipy.stats import ttest_1samp
# corrResults = np.zeros((len(taskType),len(ROIname),len(MEG),len(modelList))) 
# 2 x 4 ROI x 3 model
for taskN in range(len(taskType)):
    pValue = np.zeros((len(ROIname),len(modelList)))
    for i in range(len(ROIname)):
        for j in range(len(modelList)):
            tValue,pValue[i,j] = ttest_1samp(corrResults[taskN,i,:,j],0)
    print(pValue)

print('AllDone')
print('AllDone')