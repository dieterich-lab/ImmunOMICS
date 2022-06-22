# import scipy
from sklearn.preprocessing import LabelEncoder, minmax_scale
import pickle
import numpy as np
import pandas as pd
####*IMPORANT*: Have to do this line *before* importing tensorflow
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from collections import defaultdict
import shap
tf.compat.v1.disable_v2_behavior()

x_exp=pd.read_csv (snakemake.input[0],index_col=0)
x_exp=x_exp.loc[x_exp['condition'].isin(['Mild','Severe']),:]

model_j_e=snakemake.input[1]

out2=snakemake.output[0]
out_fig=snakemake.output[1]
out_e_txt=snakemake.output[2]
out_shap=snakemake.output[3]



label= x_exp.iloc[:,-2].values
x_exp= x_exp.drop('condition',axis=1)
x_exp= x_exp.drop('who_score',axis=1)

genes = x_exp.columns

le = LabelEncoder()
Ytest = le.fit_transform(label)

x_exp = minmax_scale(x_exp, axis = 0)




with open(model_j_e, 'rb') as b:
    model_e=pickle.load(b)



y_score2 = pd.DataFrame([])


i=0
for model in model_e.values():
    y_score2['sampling'+str(i)]=model.predict(x_exp).flatten()
    i=i+1

    
y_score2.to_csv(out2)


def confidence_interval( values):
        mean = np.mean(values)
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        bottom = max(0.0, np.percentile(values, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        top = min(1.0, np.percentile(values, p))
        return mean, bottom, top
def eval_box(metrics,tit):
    f = plt.figure()
    val_vec=list()
    bt_vec=list()
    tp_vec=list()
    x_vec=list()
    for d in metrics:
        val, bt, tp= confidence_interval(metrics[d])
        val_vec.append(val)
        bt_vec.append(val-bt)
        tp_vec.append(tp-val)
        x_vec.append(d)
    
    with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
#         plt.errorbar(x_vec, val_vec, yerr=(bt_vec, tp_vec), linestyle="None",  fmt="ob",  capsize=3,  ecolor="k")
        plt.errorbar(x_vec, val_vec, yerr=(bt_vec, tp_vec),fmt='o', 
            mfc = 'blue',
            mec = 'blue',
            ms = 4,
            mew = 3, 
            ecolor='k', 
            lw=3,
            ls = ':',
            color='blue')
        plt.title(tit)
        plt.plot([], c='k', label='CI 95%')
        plt.plot([], c='blue', label='mean')
        plt.legend(loc="lower right")
#         plt.savefig(tit, format='pdf', dpi=360)
#         plt.show()
    return f
def compute_metrics(y):
    metrics = defaultdict(list)
    for l in y.columns:
            metrics['auc'].append(skm.roc_auc_score(Ytest,y[l]))
            metrics['acc'].append(skm.accuracy_score(Ytest,y[l]>=0.5))
            metrics['f1'].append(skm.f1_score(Ytest,y[l]>=0.5))
            metrics['rec'].append(skm.recall_score(Ytest,y[l]>=0.5))
            metrics['prc'].append(skm.precision_score(Ytest,y[l]>=0.5))
            metrics['auprc'].append(skm.average_precision_score(Ytest,y[l]))
    return metrics
res_e=compute_metrics(y_score2)

#compute mean and CI 
all_met= pd.DataFrame([])
for d in res_e:
        val, bt, tp= confidence_interval(res_e[d])
        all_met[d]=[val,bt,tp]
all_met.index=['mean','lower CI','upper CI']
all_met.transpose().to_csv(out_e_txt)



#plot figures

fig7=eval_box({'Precision':res_e['prc'],'Recall':res_e['rec'],'F1-score':res_e['f1']},'Sens&Spec_CC&GE')

pp = PdfPages(out_fig)
pp.savefig(fig7)
pp.close()



#SHAP values
x_ref_exp=pd.read_csv (snakemake.input[2],index_col=0)
x_ref_exp=x_ref_exp.loc[x_ref_exp['condition'].isin(['Mild','Severe']),:]

x_ref_exp= x_ref_exp.drop('condition',axis=1)
x_ref_exp= x_ref_exp.drop('who_score',axis=1)
x_ref_exp = minmax_scale(x_ref_exp, axis = 0)


for model_joint in model_e.values():
    explainer = shap.DeepExplainer(model_joint, x_ref_exp)
    shap_values = explainer.shap_values(np.array(x_exp))
    if model_joint ==model_e[0]:
        shap_values_all_exp=shap_values[0]
    else:
        shap_values_all_exp=shap_values_all_exp+shap_values[0]
nb=len(model_e)
f1 = plt.figure()
with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
    shap.summary_plot(shap_values_all_exp/nb, plot_type= 'violin',features=np.array(x_exp)
                      , feature_names =genes,color_bar_label='Feature value',show=False)
f2 = plt.figure()

with plt.rc_context({'figure.figsize': (4, 3), 'figure.dpi':300}):
    shap.summary_plot(shap_values_all_exp/nb, plot_type= 'bar',features=np.array(x_exp)
                      , feature_names =genes,color_bar_label='Feature value',show=False)
    
pp = PdfPages(out_shap)
pp.savefig(f1)
pp.savefig(f2)
pp.close()
    
