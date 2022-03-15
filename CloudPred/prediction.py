#!/usr/bin/env python
# coding: utf-8

# # Prediction based on Linear model, GMM_class and GMM_patient

# In[1]:


import pickle
import scipy
import scipy.io
import os
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sklearn.model_selection as sks
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import sklearn.metrics as skm
import collections
from sklearn.model_selection import KFold
import torch
import traceback
import random
import pathlib
import sklearn.mixture
import math
from lib import *


# In[2]:


def get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers):
            print("get data")
            df = markers.loc[markers["cluster"].isin(set_clusters),:]
            feat_tab = df.groupby('cluster')
            df2= feat_tab.apply(lambda x: x.sort_values(["avg_log2FC"], ascending=False)).reset_index(drop=True)
            feat=df2.groupby('cluster').head(n)
            print("data got")
            idx_te= np.where(adata1_c.isin (feat.gene.values))[0] 
            idx_tr= np.where(adata_c.isin (adata1_c[idx_te]))[0]            

#             markers = ['HLA-DRA','HLA-DRB1','LYZ','CST3','TYROBP','AP1S2','CSTA','FCN1','MS4A6A','LST1','CYBB','CTSS','DUSP6','IL1B','SGK1','KLF4','CLEC7A','ATP2B1-AS1','MARCKS','SAT1','MYADM','IFI27','IFITM3','ISG15','APOBEC3A','IFI6','TNFSF10','MT2A','MX1','IFIT3','MNDA','S100A12','S100A9','S100A8','MAFB','VCAN','PLBD1','CXCL8','RNASE2','FCGR3A','MS4A7','CDKN1C','AIF1','COTL1','FCER1G','C1QA','RHOC','FCGR3B','IFITM2','NAMPT','G0S2','PROK2','CMTM2','BASP1','BCL2A1','SLC25A37','DEFA3','LTF','LCN2','CAMP','RETN','DEFA4','CD24','PGLYRP1','OLFM4']
#             idx_tr= np.where(adata_c.isin (markers))[0]
#             idx_te= np.where(adata1_c.isin (markers))[0]

            Xtrain = list(map(lambda x: (x[0][:,idx_tr], *x[1:]), Xtrain))
            Xtest = list(map(lambda x: (x[0][:,idx_te], *x[1:]), Xtest))
            Xtrain = list(map(lambda x: (x[0].todense(), *x[1:]), Xtrain))
            Xtest  = list(map(lambda x: (x[0].todense(), *x[1:]), Xtest))    
            return Xtrain,Xtest


# In[3]:


def linear(Xtrain, Xtest, seed, ite):
    print("start linear")
    torch.manual_seed(seed)
    X_train=Xtrain
    kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
    for elem in kf:
                        train_index, test_index=elem
                        Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
                        print("##Train_75%")
                        torch.manual_seed(seed)
                        linear = torch.nn.Sequential(Aggregator(), Linear(Xtrain[0][0].shape[1], len(state)))
                        model, res = train_classifier(Xtrain, [], [], linear, eta=1e-3, iterations=ite, state=state, regression=False,path=path+"/"+label)
                        print("##Test_25%")
                        model, res = train_classifier([], Xvalid, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=False,path=path+"/"+label+"/"+str(n)+"_")
                        print('##Test')
                        model, res = train_classifier([], Xtest, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=False,path=path+"/"+label+"/"+str(n)+"_")
                        
    Xtrain=X_train
    linear = torch.nn.Sequential(Aggregator(), Linear(Xtrain[0][0].shape[1], len(state)))
    print('##Trian_100%')
    model, res = train_classifier(Xtrain, [], [], linear, eta=1e-3, iterations=ite, state=state, regression=False,path=path+"/"+label+"/"+str(n)+"_")
    print('##Test')
    model, res = train_classifier([], Xtest, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=False,path=path+"/"+label+"/"+str(n)+"_")
                        


# In[11]:


def GMM_class(Xtrain, Xtest, seed, set_centers):
    print("start generative with seed: "+str(seed))
    best_model = None
    best_score = -float("inf")
    X_train = Xtrain
    for centers in set_centers:
                    kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
                    for elem in kf:
                        train_index, test_index=elem
                        Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
                        model = train_class(Xtrain, centers,seed)
                        res = eval_class(model, Xtrain,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")
                        res = eval_class(model, Xvalid,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")

                        if (res["accuracy"] > best_score) or (res["accuracy"] == best_score and res["auc"] > best_score_auc):
                            best_model = model
                            best_score = res["accuracy"]
                            best_score_auc = res["auc"]
                            res_=res
                            best_centers = centers

    with open(path+'/gmm_class_'+str(seed)+"_"+str(best_centers)+"_"+label+"_"+str(n), 'wb') as fp:
                     pickle.dump(best_model, fp)             
    
    print("##Best Validation")
    print(res_)
    print('best center: ' + str(best_centers))
    print("##Test")
    res =eval_class(best_model, Xtest,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")
    print(res)


# In[12]:


def GMM_patient(Xtrain, Xtest, seed, set_centers):
    print("start generative patient with seed: "+str(seed))
    best_model = None
    best_score = -float("inf")
    X_train = Xtrain
    for centers in set_centers:
                    kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
                    for elem in kf:
                        train_index, test_index=elem
                        Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
                        model = train_patient(Xtrain, centers,seed)
                        res = eval_patient(model,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")
                        res = eval_patient(model,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")

                        if (res["accuracy"] > best_score) or (res["accuracy"] == best_score and res["auc"] > best_score_auc):
                            best_model = model
                            best_score = res["accuracy"]
                            best_score_auc = res["auc"]
                            res_=res
                            best_centers = centers

    with open(path+'/gmm_patient_'+str(seed)+"_"+str(best_centers)+"_"+label+"_"+str(n), 'wb') as fp:
                     pickle.dump(best_model, fp)                    
    
    print("##Best Validation")
    print(res_)
    print('best center: ' + str(best_centers))
    print("##Test")
    res =eval_patient(best_model, Xtest,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")
    print(res)


# ## 1.Load data (all cells, Monocytes, Monocytes+neutrophils, Neutrophils)

# In[6]:


data = collections.defaultdict(list)
path= "../../2cohorts/chr2chr1/"
files= ['Xtest','Xall','Ctest','Call','state']
ext= ['_who']
for fl in files:
    for ex in ext:
        car= fl+ex
        with open(path+fl+ex+".pkl", "rb") as f:
            buf = pickle.load(f)
        if fl ==  'Xall':
            leng= len(buf)-1
            bu = buf[0]
            for i in range(leng):
                bu=np.concatenate([bu,buf[i+1]])
            buf = bu
        data[car]=buf


# ## Load marker genes

# In[7]:


markers= pd.read_csv('../scripts/marker_cohort2')
markers["avg_log2FC"] = np.abs(markers["avg_log2FC"])


# ## 2.Prediction based on Monocytes 

# ### 2.1.Using top 50 genes per cluster

# In[8]:


# label= 'mono'
# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# set_clusters=[7,11,3,4,6]
# n = 50
# adata_c = data['Call_mono']
# adata1_c = data['Ctest_mono']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
# state = data['state']


# # ### 2.1.1.Linear model

# # In[23]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# # ### 2.1.2.GMM_class

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 2.1.3.GMM patient 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ### 2.2.Using top 20 genes per cluster

# # In[36]:

# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 2.2.1.Linear model

# # In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)



# # ### 2.2.2.GMM class 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 2.2.3.GMM patient 

# # In[36]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ### 2.3.Using top 100 genes per cluster

# # In[36]:

# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 2.3.1.Linear model

# # In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)



# # ### 2.3.2.GMM class 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 2.3.3.GMM patient 

# # In[36]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ## 3.Prediction based on Monocytes+Neutrophils 

# # ### 3.1.Using top 50 genes per cluster

# # In[8]:


# label='mono_neu'
# Xtrain = data['Xall_mono_neu']
# Xtest = data['Xtest_mono_neu']
# set_clusters=[7,11,3,4,6,9,14]
# n = 50
# adata_c = data['Call_mono_neu']
# adata1_c = data['Ctest_mono_neu']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
# state = data['state']


# # ### 3.1.1.Linear model

# # In[ ]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# # ### 3.1.2.GMM_class

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 3.1.3.GMM patient 

# # In[34]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ### 3.2.Using top 20 genes per cluster

# # In[36]:

# Xtrain = data['Xall_mono_neu']
# Xtest = data['Xtest_mono_neu']
# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 3.2.1.Linear model

# # In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# # ### 3.2.2.GMM class 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 3.2.3.GMM patient 

# # In[36]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ### 3.3.Using top 100 genes per cluster

# # In[36]:


# Xtrain = data['Xall_mono_neu']
# Xtest = data['Xtest_mono_neu']
# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 3.3.1.Linear model

# # In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# # ### 3.3.2.GMM class 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 3.3.3.GMM patient 

# # In[36]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ## 4.Prediction based on Neutrophils 

# # ### 4.1.Using top 50 genes per cluster

# # In[8]:


# label='neu'
# Xtrain = data['Xall_neu']
# Xtest = data['Xtest_neu']
# set_clusters=[9,14]
# n = 50
# adata_c = data['Call_neu']
# adata1_c = data['Ctest_neu']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
# state = data['state']


# # ### 4.1.1.Linear model

# # In[ ]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# ### 4.1.2.GMM_class

# In[ ]:


# for seed in [0,42,10,1234,4321]:
#     GMM_class(Xtrain, Xtest, seed, [21,5])


# ### 4.1.3.GMM patient 

# In[34]:


# for seed in [0,42,10,1234,4321]:
#     GMM_patient(Xtrain, Xtest, seed, [21,5])


# ### 4.2.Using top 20 genes per cluster

# In[36]:


# Xtrain = data['Xall_neu']
# Xtest = data['Xtest_neu']
# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 4.2.1.Linear model

# # In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)



# # ### 4.2.2.GMM class 

# # In[ ]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_class(Xtrain, Xtest, seed, [21,5])


# # ### 4.2.3.GMM patient 

# # In[36]:


# # for seed in [0,42,10,1234,4321]:
# #     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # ### 4.3.Using top 100 genes per cluster

# # In[36]:


# Xtrain = data['Xall_neu']
# Xtest = data['Xtest_neu']
# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# ### 4.3.1.Linear model

# In[37]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)



# ### 4.3.2.GMM class 

# In[ ]:


# for seed in [0,42,10,1234,4321]:
#     GMM_class(Xtrain, Xtest, seed, [21,5])


# ### 4.3.3.GMM patient 

# In[36]:


# for seed in [0,42,10,1234,4321]:
#     GMM_patient(Xtrain, Xtest, seed, [21,5])


# ## 5.Prediction based on All cells 

# ### 5.1.Using top 50 genes per cluster

# In[8]:


label='who'
Xtrain = data['Xall_who']
Xtest = data['Xtest_who']
set_clusters=np.unique(markers["cluster"])
n = 50
adata_c = data['Call_who']
adata1_c = data['Ctest_who']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
state = data['state_who']


# ### 5.1.1.Linear model

# In[ ]:

# print(label)
# print(n)
# for seed in [42]:
#     linear(Xtrain, Xtest, seed, 1000)


# ### 5.1.2.GMM_class

# In[ ]:


# for seed in [0,42,10,1234,4321]:
#     GMM_class(Xtrain, Xtest, seed, [21,5])


# ### 5.1.3.GMM patient 

# In[34]:


# for seed in [0,42,10,1234,4321]:
#     GMM_patient(Xtrain, Xtest, seed, [21,5])


# ### 5.2.Using top 20 genes per cluster

# In[36]:


Xtrain = data['Xall_who']
Xtest = data['Xtest_who']
n = 20
Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# ### 5.2.1.Linear model

# In[37]:

print(label)
print(n)
for seed in [42]:
    linear(Xtrain, Xtest, seed, 1000)



# ### 5.2.2.GMM class 

# In[ ]:


# for seed in [0,42,10,1234,4321]:
#     GMM_class(Xtrain, Xtest, seed, [21,5])


# ### 5.2.3.GMM patient 

# In[36]:


# for seed in [0,42,10,1234,4321]:
#     GMM_patient(Xtrain, Xtest, seed, [21,5])


# ### 5.3.Using top 100 genes per cluster

# In[36]:


Xtrain = data['Xall']
Xtest = data['Xtest']
n = 100
Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# ### 5.3.1.Linear model

# In[37]:

print(label)
print(n)
for seed in [42]:
    linear(Xtrain, Xtest, seed, 1000)



# ### 5.3.2.GMM class 

# In[ ]:


# for seed in [0,42,10,1234,4321]:
#     GMM_class(Xtrain, Xtest, seed, [21,5])


# ### 5.3.3.GMM patient 

# In[36]:


# for seed in [0,42,10,1234,4321]:
#     GMM_patient(Xtrain, Xtest, seed, [21,5])


# # Test on Cohort1

# In[49]:


# with open("../../2cohorts_markers/chr2chr1/Xtest__.pkl", "rb") as f:
#                 Xtest = pickle.load(f)
# with open("../../2cohorts_markers/chr2chr1/Xall.pkl", "rb") as f:
#                 Xall = pickle.load(f)
# with open("../../2cohorts_markers/chr2chr1/Ctest__.pkl", "rb") as f:
#                 adata1_c = pickle.load(f)
# with open("../../2cohorts_markers/chr2chr1/Call__.pkl", "rb") as f:
#                 adata_c = pickle.load(f)
                
# Xtrain=np.concatenate([Xall[0],Xall[1]])
            
# df= pd.read_csv('../scripts/marker_cohort2')
# df["avg_log2FC"] = np.abs(df["avg_log2FC"])
# # df = df.loc[df["cluster"].isin([7,11,3,4,6]),:]
# feat_tab = df.groupby('cluster')
# df2= feat_tab.apply(lambda x: x.sort_values(["avg_log2FC"], ascending=False)).reset_index(drop=True)
# feat=df2.groupby('cluster').head(20)
# idx_te= np.where(adata1_c.isin (feat.gene.values))[0] 
# idx_tr= np.where(adata_c.isin (adata1_c[idx_te]))[0]            

# #             markers = ['HLA-DRA','HLA-DRB1','LYZ','CST3','TYROBP','AP1S2','CSTA','FCN1','MS4A6A','LST1','CYBB','CTSS','DUSP6','IL1B','SGK1','KLF4','CLEC7A','ATP2B1-AS1','MARCKS','SAT1','MYADM','IFI27','IFITM3','ISG15','APOBEC3A','IFI6','TNFSF10','MT2A','MX1','IFIT3','MNDA','S100A12','S100A9','S100A8','MAFB','VCAN','PLBD1','CXCL8','RNASE2','FCGR3A','MS4A7','CDKN1C','AIF1','COTL1','FCER1G','C1QA','RHOC','FCGR3B','IFITM2','NAMPT','G0S2','PROK2','CMTM2','BASP1','BCL2A1','SLC25A37','DEFA3','LTF','LCN2','CAMP','RETN','DEFA4','CD24','PGLYRP1','OLFM4']
# #             idx_tr= np.where(adata_c.isin (markers))[0]
# #             idx_te= np.where(adata1_c.isin (markers))[0]

# Xtrain = list(map(lambda x: (x[0][:,idx_tr], *x[1:]), Xtrain))
# Xtest = list(map(lambda x: (x[0][:,idx_te], *x[1:]), Xtest))
# Xtrain = list(map(lambda x: (x[0].todense(), *x[1:]), Xtrain))
# Xtest  = list(map(lambda x: (x[0].todense(), *x[1:]), Xtest))
# X_train= Xtrain


# # In[53]:


# for seed in [42]:
#     kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
#     for elem in kf:
#                         train_index, test_index=elem
#                         Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
#                         print(len(Xvalid))
#                         print("start")
#                         torch.manual_seed(seed)
#                         linear = torch.nn.Sequential(Aggregator(), Linear(Xtrain[0][0].shape[1], len(state)))
#                         model, res = train_classifier(Xtrain, [], [], linear, eta=1e-3, iterations=1000, state=state, regression=False)
#                         model, res = train_classifier([], Xvalid, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=False, suff='linear')
#                         print(res)
#                         print('test')
#                         model, res = train_classifier([], Xtest, [], model, regularize=None, iterations=1, eta=0, stochastic=True, regression=False, suff='linear')
#                         print(res)                        

