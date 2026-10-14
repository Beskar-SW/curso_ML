"""Centralized metrics; undefined clustering metrics return None."""
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
 confusion_matrix,balanced_accuracy_score,roc_auc_score,average_precision_score,
 silhouette_score,davies_bouldin_score,calinski_harabasz_score,
 adjusted_rand_score,normalized_mutual_info_score)
def classification(y_true,y_pred,scores=None):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    out={"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,zero_division=0),
         "recall":recall_score(y_true,y_pred,zero_division=0),"f1":f1_score(y_true,y_pred,zero_division=0),
         "specificity":tn/(tn+fp) if tn+fp else None,
         "balanced_accuracy":balanced_accuracy_score(y_true,y_pred)}
    if scores is not None and len(np.unique(y_true))==2:
        out["roc_auc"]=roc_auc_score(y_true,scores)
        out["average_precision"]=average_precision_score(y_true,scores)
    return out
def clustering(X,labels,y_true=None):
    labels=np.asarray(labels); valid=labels!=-1
    groups=np.unique(labels[valid])
    out={"n_clusters":len(groups),"noise_fraction":float(np.mean(~valid))}
    if len(groups)>=2 and valid.sum()>len(groups):
        out.update(silhouette=silhouette_score(X[valid],labels[valid]),
                   davies_bouldin=davies_bouldin_score(X[valid],labels[valid]),
                   calinski_harabasz=calinski_harabasz_score(X[valid],labels[valid]))
    else: out.update(silhouette=None,davies_bouldin=None,calinski_harabasz=None)
    if y_true is not None:
        out.update(ari=adjusted_rand_score(y_true,labels),nmi=normalized_mutual_info_score(y_true,labels))
    return out
