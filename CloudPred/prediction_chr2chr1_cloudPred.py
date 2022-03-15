#!/usr/bin/env python
# coding: utf-8

# # Prediction based on CloudPred

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
from lib_cloudpred import *


# In[2]:


def get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers):
            df = markers.loc[markers["cluster"].isin(set_clusters),:]
            feat_tab = df.groupby('cluster')
            df2= feat_tab.apply(lambda x: x.sort_values(["avg_log2FC"], ascending=False)).reset_index(drop=True)
            feat=df2.groupby('cluster').head(n)
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


def cloud_pred(Xtrain, Xtest, seed, set_centers):
    
            best_model = None
            best_score = float("inf")
            print("start cloudpred")
            print(seed)
            for centers in set_centers:
                model, res = train(Xtrain, [], centers, regression=False, seed=seed)
                print("best score for center"+ str(res["loss"])+"_"+str( centers))
                print(res)
                if res["loss"] < best_score:
                    best_model = model
                    best_score = res["loss"]
                    best_centers = centers
                    res_ = res
            with open(path+'/cloudpred_model'+label+"_"+str(seed)+"_"+str(best_centers)+"_"+str(n), 'wb') as fp:
                 pickle.dump(best_model, fp)         
            print("##Validation")
            print(res_)
            res = eval(best_model, Xtest, regression=False,path=path+"/"+label+"/"+str(seed)+"_"+str(n)+"_")
            print("##Test")
            print(res)


# ## 1.Load data (all cells, Monocytes, Monocytes+neutrophils, Neutrophils)

# In[4]:


data = collections.defaultdict(list)
path= "../../2cohorts/chr2chr1/"
files= ['Xtest','Xall','Ctest','Call','state']
ext= ['','_mono','_mono_neu','_neu']
for fl in files:
    for ex in ext:
        car= fl+ex
        with open(path+fl+ex+".pkl", "rb") as f:
            buf = pickle.load(f)
        if fl ==  'Xall':
            buf=np.concatenate([buf[0],buf[1]])
        data[car]=buf


# ## Load marker genes

# In[5]:


markers= pd.read_csv('../scripts/marker_cohort2')
markers["avg_log2FC"] = np.abs(markers["avg_log2FC"])


# ## 2.Prediction based on Monocytes 

# ### 2.1.Using top 50 genes per cluster

# In[6]:


# label= 'mono'
# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# set_clusters=[7,11,3,4,6]
# n = 50
# adata_c = data['Call_mono']
# adata1_c = data['Ctest_mono']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
state = data['state']


# # ### 2.1.1.CloudPred model

# # In[8]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [5])


# # ### 2.2.Using top 20 genes per cluster

# # In[12]:


# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 2.2.2.CloudPred class 

# # In[13]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [5])


# # ### 2.3.Using top 100 genes per cluster

# # In[15]:


# Xtrain = data['Xall_mono']
# Xtest = data['Xtest_mono']
# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 2.3.2.GMM class 

# # In[16]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [5])


# # ## 3.Prediction based on Monocytes+Neutrophils 

# # ### 3.1.Using top 50 genes per cluster

# # In[ ]:


# label='mono_neu'
# Xtrain = data['Xall_mono_neu']
# Xtest = data['Xtest_mono_neu']
# set_clusters=[7,11,3,4,6,9,14]
# n = 50
# adata_c = data['Call_mono_neu']
# adata1_c = data['Ctest_mono_neu']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
# state = data['state']


# # ### 3.1.2.GMM_class

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [7,5])


# # ### 3.2.Using top 20 genes per cluster

# # In[ ]:


# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 3.2.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [7,5])


# # ### 3.3.Using top 100 genes per cluster

# # In[ ]:


# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 3.3.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [7,5])


# # ## 4.Prediction based on Neutrophils 

# # ### 4.1.Using top 50 genes per cluster

# # In[ ]:


# label='neu'
# Xtrain = data['Xall_neu']
# Xtest = data['Xtest_neu']
# set_clusters=[9,14]
# n = 50
# adata_c = data['Call_neu']
# adata1_c = data['Ctest_neu']
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
# state = data['state']


# # ### 4.1.2.GMM_class

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [2])


# # ### 4.2.Using top 20 genes per cluster

# # In[ ]:


# n = 20
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 4.2.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [2])


# # ### 4.3.Using top 100 genes per cluster

# # In[ ]:


# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 4.3.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [2])


# # ## 5.Prediction based on All cells 

# # ### 5.1.Using top 50 genes per cluster

# # In[17]:


label='all'
Xtrain = data['Xall']
Xtest = data['Xtest']
set_clusters=np.unique(markers["cluster"])
n = 50
adata_c = data['Call']
adata1_c = data['Ctest']
Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)
state = data['state']


# # ### 5.1.2.GMM_class

# # In[ ]:


for seed in [10,1234,4321]:
    cloud_pred(Xtrain, Xtest, seed, [21])


# ### 5.2.Using top 20 genes per cluster

# In[ ]:


# # Xtrain = data['Xall']
# # Xtest = data['Xtest']
# # n = 20
# # Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 5.2.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [21,5])


# # ### 5.3.Using top 100 genes per cluster

# # In[ ]:


# Xtrain = data['Xall']
# Xtest = data['Xtest']
# n = 100
# Xtrain,Xtest = get_data(Xtrain,Xtest,set_clusters,n,adata_c,adata1_c,markers)


# # ### 5.3.2.GMM class 

# # In[ ]:


# for seed in [0,42,10,1234,4321]:
#     cloud_pred(Xtrain, Xtest, seed, [21,5])


# In[ ]:





# In[ ]:





# In[ ]:




