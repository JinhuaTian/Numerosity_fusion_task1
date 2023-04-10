# -*- coding: utf-8 -*-
'''
prepare data as n subjects x 45 events&2 tasks x n times
nomorlize data? loss
compute pearson correlation for each time points 1-r
'''
import numpy as np
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from os.path import join as pj
import math
import pandas as pd

# basic info
rootDir = '/data/home/nummag01/workingdir/fusion1/MEG4/'
method = 'pearson' #'pearson' 'euclidean' 'mahalanobis' (sample>dimension)
# 'MDS_pearson.npy', MDS_euclidean.npy 
# regions, subjNumber, time points, vector
data4region = np.load(pj(rootDir, "MDS_"+ method +".npy")) 
imgNum = 45
# data4region = data4region[:,[1],:,:]

regionName = ['occipital', 'parietal','central', 'frontal']
eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/fusion1/MEG4/STI2regression.txt')
modelElement = np.zeros((imgNum*2,3)) # model elements, element dimensions
for i in range(imgNum*2):
    if i<imgNum:
        modelElement[i,0] = 0 # number task
        modelElement[i,1] = eventMatrix[i,0]
        modelElement[i,2] = eventMatrix[i,1]
    elif i>=imgNum:
        j = i-imgNum
        modelElement[i,0] = 1 # FA task
        modelElement[i,1] = eventMatrix[j,0]
        modelElement[i,2] = eventMatrix[j,1]
    
Xa = modelElement[0:imgNum,:]
Xb = modelElement[imgNum:,:]
Ya = np.array([[1,0,0],[0,0,0],[0,0,1]])
Yb = np.array([[1,0,0],[0,1,0],[0,0,0]])
def transRad2Deg(m):
    degree = math.radians(m)
    return degree
Ra = np.array([[1,0,0],[0,math.cos(transRad2Deg(90)),math.sin(transRad2Deg(90))],[0,-math.sin(transRad2Deg(90)),math.cos(transRad2Deg(90))]])
proj = np.array([[1,0],[0,math.cos(transRad2Deg(45))],[0,math.sin(transRad2Deg(45))]])

# Ra45 = np.array([[1,0,0],[0,math.cos(transRad2Deg(45)),math.sin(transRad2Deg(45))],[0,-math.sin(transRad2Deg(45)),math.cos(transRad2Deg(45))]])

modelElement_Grid = modelElement
modelElement_Orth = np.concatenate((Xa@Ya,Xb@Yb),axis=0)
modelElement_Par = np.concatenate((Xa@Ya@Ra,Xb@Yb),axis=0)
modelElement_Rotdrid = np.concatenate((Xa@Ra,Xb),axis=0)
modelElement_Num = np.concatenate((Xa@Ya,Xb@Ya),axis=0)
modelElement_Fa = np.concatenate((Xa@Yb,Xb@Yb),axis=0)
modelElement_Diag = modelElement_Grid@proj # @proj.T??? two dimension
del modelElement,Xa,Xb,Ya,Yb,Ra,proj

labelNum = imgNum*2 #stiNum * 2tasks
indexNum = 0
vectorLength = 0
for x in range(labelNum):
    for y in range(x+1,labelNum):   
        vectorLength = vectorLength+1
# vectorLength = 1225 # 4005
subjNum = 2
modelName = ['Grid','Orthogonal','Parallel','Rotated Grid','Number','Field area','Diagonal'] #,'Diagonal'

#plot model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

models = [modelElement_Grid,modelElement_Orth,modelElement_Par, modelElement_Rotdrid, modelElement_Num, modelElement_Fa] #, modelElement_Diag
printModel = False
if printModel == True:
    modelNum = 1
    for modelNum in range(len(models)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(labelNum):
            ax.scatter(models[modelNum][i,0],models[modelNum][i,1],models[modelNum][i,2])
        ax.set_xlabel('Task axis')
        ax.set_ylabel('Field area axis')
        ax.set_zlabel('Number axis')
        # set the axis scale  https://blog.csdn.net/weixin_44520259/article/details/89917026
        xLocator=MultipleLocator(1)
        yLocator = MultipleLocator(1)
        zLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(xLocator)
        ax.yaxis.set_major_locator(yLocator)
        ax.zaxis.set_major_locator(zLocator)        
        # plt.ylim()
        # plt.xlim()
        ax.grid(False) # remove the background figure
        plt.title(modelName[modelNum]+' model')
        plt.tight_layout()
        fig.savefig('RDMmodel'+str(modelNum+1)+'.png')
        plt.show()

factorX = np.zeros((vectorLength,7))
# 70 140ms, 120 240ms, 150 300ms
# factorY = data[:,70,:]
for x in range(labelNum):
    for y in range(x+1,labelNum):    
        factorX[indexNum,0] = np.linalg.norm(modelElement_Grid[x,:]-modelElement_Grid[y,:])
        factorX[indexNum,1] = np.linalg.norm(modelElement_Orth[x,:]-modelElement_Orth[y,:])
        factorX[indexNum,2] = np.linalg.norm(modelElement_Par[x,:]-modelElement_Par[y,:])
        factorX[indexNum,3] = np.linalg.norm(modelElement_Rotdrid[x,:]-modelElement_Rotdrid[y,:])
        factorX[indexNum,4] = np.linalg.norm(modelElement_Num[x,:]-modelElement_Num[y,:])
        factorX[indexNum,5] = np.linalg.norm(modelElement_Fa[x,:]-modelElement_Fa[y,:])
        factorX[indexNum,6] = np.linalg.norm(modelElement_Diag[x,:]-modelElement_Diag[y,:])
        indexNum = indexNum + 1
from sklearn import preprocessing as preprocessing
# z-score regressor
for i in range(factorX.shape[1]):
    factorX[:,i] = preprocessing.scale(factorX[:,i])

# select part of models
modelSelect = [0,1,3,4,5]  # [0,1,2,3,4,5,6] # [0,1,3,4,5] # [0,1,3]
modelName = [modelName[i] for i in modelSelect]
factorX = factorX[:,modelSelect]

data = data4region
# if selectModel == True:
### fit model for each brain region and each time point
params = np.zeros((subjNum,data.shape[1],len(modelName)))
# use linear regression
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso as Ls
Rmethod = 'LR'

data = data4region
### fit model for each brain region and each time point
params = np.zeros((subjNum,data.shape[1],len(modelName)))
if Rmethod == 'LR':
    # use linear regression    
    for t in range(data.shape[1]):
        factorY = data[:,t,:]    
        for i in range(subjNum):
            model = LR().fit(factorX,factorY[i]) #生成模型
            params[i,t,:] = model.coef_
elif Rmethod == 'LS':
    #use Lasso regression    
    for t in range(data.shape[1]):
        factorY = data[:,t,:]    
        for i in range(subjNum):
            model = Ls(alpha=1).fit(factorX,factorY[i]) # Ls(alpha=1).fit(factorX,factorY[i])
            params[i,t,:] = model.coef_

# import statsmodels.api as sm
# for t in range(data.shape[1]):
#     factorY = data[:,t,:]    
#     for i in range(subjNum):
#         model = sm.OLS(factorY[i], factorX) #生成模型
#         result = model.fit() #模型拟合
#         params[i,t,:] = result.params
#         #print(result.summary())

paramsAvg = np.average(params, axis=(0))
ax = plt.figure(figsize=(9, 6), dpi=100)
for modelNum in range(len(modelName)):
    plt.plot(range(0,702,2),paramsAvg[:,modelNum],label=modelName[modelNum])
plt.title('Time course of Model parameters changes')
plt.legend()
plt.show()

for t in range(data.shape[1]):
    factorY = data[:,t,:]    
    for i in range(subjNum):
        model = LR().fit(factorX,factorY[i]) #生成模型
        params[i,t,:] = model.coef_

# #use Lasso regression
# from sklearn.linear_model import Lasso as Ls
# for t in range(data.shape[1]):
#     factorY = data[:,t,:]    
#     for i in range(subjNum):
#         model = Ls(alpha=0.005).fit(factorX,factorY[i]) #生成模型
#         params[i,t,:] = model.coef_
 
# import statsmodels.api as sm
# for t in range(data.shape[1]):
#     factorY = data[:,t,:]    
#     for i in range(subjNum):
#         model = sm.OLS(factorY[i], factorX) #生成模型
#         result = model.fit() #模型拟合
#         params[i,t,:] = result.params
#         #print(result.summary())

paramsAvg = np.average(params, axis=(0))
ax = plt.figure(figsize=(9, 6), dpi=100)
for modelNum in range(len(modelName)):
    plt.plot(range(-100,602,2),paramsAvg[:,modelNum],label=modelName[modelNum])
plt.title('Time course of Model parameters changes')
plt.legend()
plt.show()

###

# fit model at ONLY ONE time point
import statsmodels.api as sm
#fotorX = factorX.transpose(1,0)
factorY = data[:,70,:]
params = np.zeros((subjNum,len(modelName)))
for i in range(subjNum):
    model = sm.OLS(factorY[i], factorX) #生成模型
    result = model.fit() #模型拟合
    params[i,:] = result.params
    print(result.summary())

# !!! shuffle along each column
def shuffleMatrix(factorX):
    factorShuffle = np.zeros((factorX.shape[0],factorX.shape[1]))
    for i in range(factorX.shape[1]):
        factorShuffle[:,i] = np.random.permutation(factorX[:,i])
    return factorShuffle


# shuffle data 1000 times and check significance
shuffleTimes = 1000
shuffleParam = np.zeros((subjNum,shuffleTimes,7))
for i in range(subjNum):
    for j in range(shuffleTimes):
        shuffleX = shuffleMatrix(factorX.copy())
        model = sm.OLS(factorY[i], shuffleX) 
        result = model.fit() #模型拟合
        shuffleParam[i,j,:] = result.params

pValue = np.zeros((subjNum,len(modelName)))
# calculate p value
for i in range(subjNum):
    for j in range(len(modelName)):
        pValue[i,j] = np.sum(shuffleParam[i,:,j]>params[i,j])
pValue = pValue/shuffleTimes
print(pValue)

# subject, time, modelNumber 140ms and 280ms
params=params*10

params0 = params[:,70,:] # 120 ,180
# plot resultsimport pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
fig3 = plt.figure(figsize=(9,6),dpi=300)
# ,modelName[5]: params[:,5],modelName[6]: params[:,6]
dataMax = pd.DataFrame(
    {modelName[0]: params0[:,0], modelName[1]: params0[:,1], modelName[2]: params0[:,2], modelName[3]: params0[:,3],modelName[4]: params0[:,4]}) #, 'Low-level featrure': maxR[m,4]
#dataMax = rearrange(dataMax,ascending=False)    
#sns.violinplot(data=dataMax)
sns.barplot(data=dataMax)
#plt.ylim(0,0.6)
#plt.title('Correlation coefficient difference task)')
plt.ylabel('Beta estimate')
plt.legend()
plt.tight_layout()
plt.savefig(pj(rootDir, 'Regresssion140.png'))
plt.show()

params0 = params[:,140,:] # 120 ,280
# plot resultsimport pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
fig3 = plt.figure(figsize=(9,6),dpi=300)
# ,modelName[5]: params[:,5],modelName[6]: params[:,6]
dataMax = pd.DataFrame(
    {modelName[0]: params0[:,0], modelName[1]: params0[:,1], modelName[2]: params0[:,2], modelName[3]: params0[:,3],modelName[4]: params0[:,4]}) #, 'Low-level featrure': maxR[m,4]
#dataMax = rearrange(dataMax,ascending=False)    
#sns.violinplot(data=dataMax)
sns.barplot(data=dataMax)
#plt.ylim(0,0.6)
#plt.title('Correlation coefficient difference task)')
plt.ylabel('Beta estimate')
plt.legend()
plt.tight_layout()
plt.savefig(pj(rootDir, 'Regresssion280.png'))
plt.show()


# plot model correlation
from pandas.core.frame import DataFrame
corrMatrix = DataFrame(factorX.transpose(1,0))            
corr_all = corrMatrix.T.corr(method='spearman')

fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(corr_all, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
ax.set_xticklabels(['','Grid','Orthogonal','Parallel','Roated Grid','Number','Field area','Diagonal'],fontdict={'size': 10, 'color': 'black'})
ax.set_yticklabels(['','Grid','Orthogonal','Parallel','Roated Grid','Number','Field area','Diagonal'],fontdict={'size': 10, 'color': 'black'})
plt.title('Spearman correlation of seven RDMs')
plt.tight_layout()
plt.savefig(pj(rootDir, 'RegresssionModel.png'))
plt.show()

print('All Done')