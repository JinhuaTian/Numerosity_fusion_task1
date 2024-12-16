# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:44:26 2021

@author: tclem
"""
import numpy as np
# import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import join as pj
# from neurora.stuff import clusterbased_permutation_1d_1samp_1sided as clusterP
import scipy.stats as st
import matplotlib.pyplot as plt
from jhTools import clusterbased_permutation_1d_1samp_1sided, permutation_diff
# load stimulus RDM
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
saveDir = '/data/home/nummag01/workingdir/fusion1/MEG4/RSA'
# make correlation matrix
RDMName = ['Number', 'Field area', 'Density', 'Low-level Feature']
# ---------------------------------------------------
# calculate partial Spearman correlation for each RDM
# ---------------------------------------------------
'''
19 subjects: 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj022','subj023','subj024','subj025'
18 subjects: 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025'
 27 subjects: exclude subj001,subj033
 'subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj019','subj021','subj023','subj024','subj025',
 'subj027','subj028','subj029','subj031','subj032','subj034','subj035','subj037','subj038','subj040'
 25 subjects: exclude subj32 head motion, subj27 bad fMRI, subj19 low acc
 'subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj021','subj023','subj024','subj025',
 'subj028','subj029','subj031','subj033','subj034','subj037','subj038','subj040'
'''
# special subj033
subjs = ['subj001','subj002','subj003','subj004','subj005','subj006','subj007','subj008',
 'subj011','subj012','subj016','subj017','subj018','subj021','subj023','subj024','subj025',
 'subj028','subj029','subj031','subj033','subj034','subj037','subj038','subj040'] # ,'subj019'
# plot parameters
plotStat = True # if True plot significant line


####!!!!! set it False, if use new data
permuDone = False # False # True
dpi = 100

samplingRate = 200
newSamplingRate=200
RDMnum = 3 # exclude low-level feature
# make 4 dimension x 2 (r value and p value) empty matrix
subjNums, tps= len(subjs), int(samplingRate*0.8) # 3*80

modelRDM = np.load(pj(rootDir,'ModelRDM_NumFaDenLLF.npy')) # exclude LFF
RDMcorr = np.zeros((2,RDMnum,subjNums,tps))
RDMp = np.zeros((2,RDMnum,subjNums,tps))

side = 'two-sided'
partial = 'partialSpearman' #'partialSpearman','Spearman','kendall'
types = ['num','fa'] 

print('Initing')
# --------------------------------------
# plot the results for all subjects
# --------------------------------------
for j in range(len(types)):
    for subj in range(len(subjs)):
        # fileName = 'ctfRDM3x100Avgx500hz_' + subjs[subj] + types[j] + '.npy'
        # fileName = 'ctfRDM_corr_3x50Avgx500hz_' + subjs[subj] + types[j] + '.npy'
        fileName = 'ctfRDM_corr_5x100Avgx200hz_' + subjs[subj] + types[j] + '.npy'
        filePath = pj(rootDir, fileName)
        # compute partial spearman correlation, with other 2 RDM controlled
        data = np.load(filePath)  # subIndex,t,re,foldIndex,RDMindex
        # t, RDMindex = data.shape
        # data = data.reshape(t * re * foldIndex, RDMindex)
        # normalize the MEG RDM
        #  scaler = StandardScaler()
        #  data = scaler.fit_transform(data)
        # data = data.reshape(t, re, foldIndex, RDMindex)

        for tp in range(tps):
            RDMtmp = data[tp]
            pdData = pd.DataFrame(
                    {'respRDM': RDMtmp, 'RDM0': modelRDM[0], 'RDM1': modelRDM[1], 'RDM2': modelRDM[2]}) # , 'RDM3': modelRDM[3]
            # 'numRDM','isRDM','tfaRDM','denRDM','denRDM'
            modelList = ['RDM0','RDM1', 'RDM2'] #, 'RDM3'
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
            elif partial == 'kendall':
                for i in range(RDMnum):
                    corr = pg.corr(RDMtmp,pdData[modelList[i]],alternative='two-sided',method='kendall')
                    RDMcorr[j, i, subj, tp] = corr['r']
                    RDMp[j, i, subj, tp] = corr['p-val']
        del data
np.save(pj(rootDir,'rsaRDM.npy'),RDMcorr)
color = ["Red", "Blue", "Red", "Blue"]
typeName = ["Number(number task)","Field area(number task)","Number(FA task)","Field area(FA task)"]
lineStyles = ["-","--","--","-"]

# --------------------------------------
# plot the results for each subject
# --------------------------------------
plotSep=False # True # False
if plotSep == True:
    for subj in range(len(subjs)):
        count = 0
        for j in range(len(types)): # task 
            for i in range(2): # RDMnum just plot the first two RDMs
                plt.plot(np.arange(-100,700,1000/samplingRate),RDMcorr[i,j,subj,:],label=typeName[count],color=color[count],linestyle=lineStyles[count])
                # judge whether there are significant line
                # plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
                count = count+1
                
        if partial == 'partialSpearman':
            plt.ylabel('Partial spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='Spearman':
            plt.ylabel('Spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='kendall':
            plt.ylabel("Kendall's tau correlation coefficient")
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title("Time course of correlation between neural RDMs and model RDMs")# Kendall's tau 
        plt.xlabel('Time points(ms)')
        # plt.ylim(ymin,ymax)
        plt.legend(prop = {'size':12})
        plt.savefig(pj(saveDir, 'subj_'+subjs[subj]+'.png'))
        plt.show()
'''
# --------------------------------------
# plot the test results
# --------------------------------------
#plotSep=True
#if plotSep == True:
correctedP = np.zeros((4,tps)) 
newCorr = [RDMcorr[0,0],RDMcorr[0,1],RDMcorr[1,0],RDMcorr[1,1]]
newP = [RDMp[0,0],RDMp[0,1],RDMp[1,0],RDMp[1,1]]
for i in range(4):
    correctedP[i] = clusterbased_permutation_1d_1samp_1sided(newCorr[i],level=0)


fig = plt.figure(figsize=(9, 6), dpi=300)
for i in range(4): # four conditions
    avgR = np.average(newCorr[i],axis=0)
    plt.plot(np.arange(-100,700,1000/samplingRate),avgR,label=typeName[i],color=color[i],linestyle=lineStyles[i])
plt.xlabel('Time points(ms)')
if partial == 'partialSpearman':
    plt.ylabel('Partial spearman correlation coefficient')
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
elif partial =='Spearman':
    plt.ylabel('Spearman correlation coefficient')
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
elif partial =='kendall':
    plt.ylabel("Kendall's tau correlation coefficient")
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title("Time course of correlation between neural RDMs and model RDMs")# Kendall's tau
thCorr = -0.025 # np.min(RDMcorr)+0.005
# plot significant line
plt.axhline(y=0,xmin=-100,xmax= 700,color='black',linestyle="--")
for i in range(4):    
    thCorr = thCorr -0.005
    correctedP[i][(correctedP[i]==0)]=None
    correctedP[i][(correctedP[i]==1)]=thCorr
    plt.plot((np.arange(-50,tps-50))/newSamplingRate*1000,correctedP[i],color=color[i]) 

plt.ylim(-0.05,0.2)
plt.legend()
plt.savefig(pj(rootDir, 'GroupRSA_multifeature4'+partial+'.png'))
plt.show()
'''

ymin,ymax = -0.1, 0.2 #-0.1, 0.4#-0.05, 0.2

fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
ax.spines[['top', 'right']].set_visible(False)
# thCorr = -0.01 # np.min(RDMcorr)+0.005 # np.min(RDMcorr)+0.005
thCorr = -0.045 # np.min(RDMcorr)+0.005

if plotStat == False:
    # cluster-based permutation,>0
    # plot the results
    # color = ["Red", "Blue", "Gray", "Purple", "Green", "Orange", 'brown']
    # typeName = ["number","field area"]
    
    #fig = plt.figure(figsize=(9, 6), dpi=dpi)

    # lineStyles = ["-","--"]
    count = 0
    for j in range(len(types)):    
        avgR = np.average(RDMcorr[j,:],axis=1)
        for i in range(2): #RDMnum just plot the first two RDMs
            plt.plot(np.arange(-100,700,1000/samplingRate),avgR[i,:],label=typeName[count],color=color[count],linestyle=lineStyles[count])
            # judge whether there are significant line
            # plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
            count = count+1
        plt.xlabel('Time points(ms)')
        if partial == 'partialSpearman':
            plt.ylabel('Partial spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='Spearman':
            plt.ylabel('Spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='kendall':
            plt.ylabel("Kendall's tau correlation coefficient")
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title("Time course of correlation between neural RDMs and model RDMs")# Kendall's tau 
 
    # plt.ylim(ymin,ymax)
    plt.legend(prop = {'size':12})
    plt.savefig(pj(rootDir, 'GroupRSA_multifeature3'+partial+'.png'))
    plt.show()

 
elif plotStat == True:
    diff1 = RDMcorr[0,0,:,:]-RDMcorr[1,0,:,:]
    diff2 = RDMcorr[1,1,:,:]-RDMcorr[0,1,:,:]

    permutationFile = 'clusterRSA'+partial+'.npy'
    
    if permuDone == False:
        # cluster-based permutation,>0
        correctedDiff1 = clusterbased_permutation_1d_1samp_1sided(diff1,level=0,iter=10000)
        correctedDiff2 = clusterbased_permutation_1d_1samp_1sided(diff2,level=0,iter=10000)
        # cluster-based permutation,>0
        correctedP = np.zeros((2,RDMnum,tps)) 
        for j in range(2):
            for i in range(RDMnum): # stat for 4 rmds.
                correctedP[j,i,:] = clusterbased_permutation_1d_1samp_1sided(RDMcorr[j,i,:,:],level=0)
        fullclustedP = [correctedP,correctedDiff1,correctedDiff2]
        np.save(pj(rootDir,permutationFile), fullclustedP, allow_pickle=True, fix_imports=True)
    elif permuDone == True:
        fullclustedP = np.load(pj(rootDir,permutationFile))
        correctedP,correctedDiff1,correctedDiff2 = fullclustedP[0],fullclustedP[1],fullclustedP[2]
    #correctedP2=correctedP.copy()

    # correctedP=correctedP2.copy()
    # plot part data
    RDMcorr = RDMcorr[:,:,:,20:]
    correctedP = correctedP[:,:,20:]
    correctedDiff1=correctedDiff1[20:] 
    correctedDiff2=correctedDiff2[20:]

    # plot the results
    color = ["Red", "Blue", "Gray", "Purple", "Green", "Orange", 'brown']
    # typeName = ["Number","Field area"]
    typeName = ["Number(number task)","Field area(number task)","Number(FA task)","Field area(FA task)"]

    #lineStyles = ["-","--"]
    count = 0
    for j in range(len(types)):    
        avgR = np.average(RDMcorr[j,:],axis=1)
        for i in range(2): #RDMnum just plot the first two RDMs
            plt.plot(np.arange(0,700,1000/samplingRate),avgR[i,:],label=typeName[count],color=color[i],linestyle=lineStyles[count])
            # judge whether there are significant line
            # plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
            count = count+1
        plt.xlabel('Time points(ms)')
        if partial == 'partialSpearman':
            plt.ylabel('Partial spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='Spearman':
            plt.ylabel('Spearman correlation coefficient')
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
        elif partial =='kendall':
            plt.ylabel("Kendall's tau correlation coefficient")
            # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
            plt.title("Time course of correlation between neural RDMs and model RDMs")# Kendall's tau 
            
        # plot significant line
        for i in range(2):    
            thCorr = thCorr -0.003
            correctedP[j,i][(correctedP[j,i]==0)]=None
            correctedP[j,i][(correctedP[j,i]==1)]=thCorr
            plt.plot((np.arange(0,tps-newSamplingRate*0.1))/newSamplingRate*1000,correctedP[j,i,:],color=color[i],linestyle=lineStyles[j]) 
    # plot corrected diff
    # thCorr = thCorr -0.003
    # correctedDiff1[correctedDiff1==0]=None
    # correctedDiff1[correctedDiff1==1]=thCorr
    # plt.plot((np.arange(0,tps-100))/newSamplingRate*1000,correctedDiff1,label='Number effect',color='black',linestyle='-') 
    # thCorr = thCorr -0.003
    # correctedDiff2[correctedDiff2==0]=None
    # correctedDiff2[correctedDiff2==1]=thCorr
    # plt.plot((np.arange(0,tps-100))/newSamplingRate*1000,correctedDiff2,label='FA effect',color='gray',linestyle='-') 


    #plt.ylim(ymin,ymax)
    plt.legend()
    plt.savefig(pj(rootDir, 'GroupRSA_multifeature3'+partial+'.png'))
    plt.show()

fig = plt.figure(figsize=(9, 6), dpi=dpi)
thCorr = -0.02
numDiff = np.average(RDMcorr[0,0,:,:],axis=0) - np.average(RDMcorr[1,0,:,:],axis=0)
faDiff = np.average(RDMcorr[1,1,:,:],axis=0) - np.average(RDMcorr[0,1,:,:],axis=0)
plt.plot((np.arange(0,tps-newSamplingRate*0.1))/newSamplingRate*1000,numDiff,label='Number task effect',color='red',linestyle='-') 
plt.plot((np.arange(0,tps-newSamplingRate*0.1))/newSamplingRate*1000,faDiff,label='Field area task effect',color='blue',linestyle='-') 

correctedDiff1[correctedDiff1==0]=None
correctedDiff1[correctedDiff1==1]=thCorr
plt.plot((np.arange(0,tps-newSamplingRate*0.1))/newSamplingRate*1000,correctedDiff1,color='red',linestyle='-')  #,label='Number effect'
thCorr = thCorr -0.003
correctedDiff2[correctedDiff2==0]=None
correctedDiff2[correctedDiff2==1]=thCorr
plt.plot((np.arange(0,tps-newSamplingRate*0.1))/newSamplingRate*1000,correctedDiff2,color='blue',linestyle='-') #,label='FA effect'
plt.axhline(y=0,color='gray')
plt.ylim(-0.04,0.08)
plt.legend()
plt.savefig(pj(rootDir, 'GroupRSA_diff_'+partial+'.png'))
plt.show()

# peak static
peakLabels = ["Num-Num","Num-FA","FA-Num","FA-FA"]
RDMcorrPeak = RDMcorr[:,:2,:,23].copy()
# RDMcorrPeak = np.average(RDMcorr[:,:2,:,23].copy(),axis=3) # 116ms,102-132ms
RDMcorrPeak = RDMcorrPeak.reshape(4, 25)

from scipy import stats
# 1. ??????????????
model1 = RDMcorrPeak[0, :]  # ?????
model2 = RDMcorrPeak[1, :]  # ?????
model3 = RDMcorrPeak[2, :]  # ?????
model4 = RDMcorrPeak[3, :]  # ?????

# 2. ?????????? (model1 - model2)
diff1 = model1 - model2

# ?????????? (model3 - model4)
diff2 = model3 - model4

# 3. ???? - ???? (?? t ??)
t_stat_diff1, p_value_diff1 = stats.ttest_rel(model1, model2)  # ?? t ??
t_stat_diff2, p_value_diff2 = stats.ttest_rel(model3, model4)  # ?? t ??

# 4. ????? - ????????????
plt.figure(figsize=(10, 6))

# ???????????
plt.boxplot([model1, model2], positions=[1, 2], widths=0.6, patch_artist=True)

# ???????????
plt.boxplot([model3, model4], positions=[4, 5], widths=0.6, patch_artist=True)

# ????????? (model1 - model2) ????
plt.boxplot(diff1, positions=[3], widths=0.6, patch_artist=True)

# ????????? (model3 - model4) ????
plt.boxplot(diff2, positions=[6], widths=0.6, patch_artist=True)

# ??????p???0.05?
if p_value_diff1 < 0.05:
    plt.text(2.5, np.max([model1.max(), model2.max()]), '*', fontsize=20, ha='center', va='bottom')
if p_value_diff2 < 0.05:
    plt.text(5.5, np.max([model3.max(), model4.max()]), '*', fontsize=20, ha='center', va='bottom')

# ???????
plt.xticks([1, 2, 3, 4, 5, 6], ['Model 1', 'Model 2', 'Model 1 - Model 2', 'Model 3', 'Model 4', 'Model 3 - Model 4'])
plt.ylabel('RDM Peak Correlation')
plt.title('Model Comparison and Differences')
plt.tight_layout()

# ????
plt.show()


print('All Done')
print('All Done')