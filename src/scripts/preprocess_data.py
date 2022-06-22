import scipy
import scanpy as sc
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
    
cell_abundance=snakemake.input[0]
pseudo_bulk=snakemake.input[1]
foldchang=snakemake.input[2]
out1=snakemake.output[0]
ou2=snakemake.output[1]
yf=snakemake.output[2]


cell = sc.read_h5ad(cell_abundance)
exp = sc.read_h5ad(pseudo_bulk)
cell= cell[exp.obs['sampleID'].values,:]

x_exp=pd.DataFrame(exp.raw.X)
x_cell=pd.DataFrame(cell.raw.X.todense())

x_exp.index=exp.obs.sampleID
x_cell.index=cell.obs.sampleID


x_exp.columns=exp.var.index

x_cell.columns=cell.var.index

#add missing celltypes
x_cell=x_cell.drop(['Doublet','Eryth'],axis=1)


x_cell=x_cell.reindex(sorted(x_cell.columns), axis=1)

x_que_cell_stan= x_que_cell_stan.div(x_que_cell_stan.sum(axis=1), axis=0)



markers= pd.read_csv(foldchange)

markers["avg_log2FC"] = np.abs(markers["avg_log2FC"])
top_n = 15
feat_tab = markers
feat_tab= feat_tab.sort_values(["avg_log2FC"], ascending=False)
selected_genes=feat_tab.head(top_n)
selected_genes["gene"] = selected_genes.index
selected_genes.gene.values



x_exp = x_exp.loc[:,selected_genes.gene.values]
x_exp = minmax_scale(x_exp, axis = 0)



label_stan= cell.obs[['sampleID','condition']].drop_duplicates()
le = LabelEncoder()
ylabel = le.fit_transform(label_bonn['condition'].values)

x_exp.to_pickle(ou1)
x_cell.to_pickle(out2)
with open(yf, 'rb') as f:
    pickle.dump(ylabel,f)
