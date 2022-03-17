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
import copy
import sys

def train_classifier(Xtrain, Xvalid, Xtest, classifier, regularize=None,
                     eta=1e-8, iterations=3, stochastic=True, cuda=False, state=None, regression=False,path=''):

#     logger = logging.getLogger(__name__)

    if torch.cuda.is_available() and cuda:
        classifier.cuda()

    if regression:
        criterion = torch.nn.modules.MSELoss()
    else:
        criterion = torch.nn.modules.CrossEntropyLoss()
#     if len(Xtrain)>0:
#         kf = KFold(n_splits=5)
#         X_train = Xtrain
#         res_kfold=kf.split(X_train)
#         for train_index, test_index in res_kfold: 
#             Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
#             pass
    batch = {"train": [None for _ in range(len(Xtrain))],
             "valid": [None for _ in range(len(Xvalid))],
             "test": [None for _ in range(len(Xtest))]}

    optimizer = torch.optim.SGD(classifier.parameters(), lr=eta, momentum=0.9)

    log = []
    best_res = {"accuracy": -float("inf")}
    best_res = {"loss": float("inf")}
    best_model = None


    for iteration in range(iterations):
#         logger.debug("Iteration #" + str(iteration + 1) + ":")
        # logger.debug(list(classifier.parameters()))
#         if len(Xtrain)>0:
#             kf = KFold(n_splits=5).split(X_train)
#         else:
#             kf = ['e']
#         for elem in kf: 
#             if len(Xtrain)>0:
#                 train_index, test_index=elem
#                 Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
        print(iteration)
        for dataset in ["train", "valid"] + (["test"] if (iteration == (iterations - 1)) else []):
                if dataset == "train":
#                     logger.debug("    Training:")
                    X = Xtrain
                elif dataset == "valid":
#                     logger.debug("    Validation:")
                    X = Xvalid
                elif dataset == "test":
#                     logger.debug("    Testing:")
                    X = Xtest
                else:
                    raise NotImplementedError()
                n = len(X)
                total = 0.
                correct = 0
                prob = 0.
                loss = 0.
                y_score = []
                pred_=[]
                y_true = []
                reg = 0.
                for (start, (x, y, *_)) in enumerate(X):
                    if batch[dataset][start] is None:
                        if isinstance(x, torch.Tensor):
                            pass
                        elif isinstance(x, np.ndarray):
                            x = torch.Tensor(x)
                        else:
                            x = x.tocoo()
                            v = torch.Tensor(x.data)
                            i = torch.LongTensor([x.row, x.col])
                            x = torch.sparse.FloatTensor(i, v, x.shape)

                        if dataset != "test":
                            if regression:
                                y = torch.FloatTensor([y])
                            else:
                                y = torch.LongTensor([y])

                        if torch.cuda.is_available() and cuda:
                            x = x.cuda()
                            if dataset != "test":
                                y = y.cuda()

                        batch[dataset][start] = (x, y)
                    else:
                        x, y = batch[dataset][start]

#                     t = time.time()

                    z = classifier(x)
                    if regression:
                        if len(z.shape) == 2:
                            if z.shape[1] == 1:
                                z = z[:, 0]
                            elif z.shape[1] == 2:
                                z = z[:, 1]
                        y_score.append(z[0].detach().numpy().item())
                    else:
                        y_score.append((z[0, 1] - z[0, 0]).detach().numpy().item())
                    y_true.append(y.detach().numpy().item())
                    pred = torch.argmax(z)
    #                 print(pred)
                    pred_.append(torch.argmax(z))
                    if dataset != "test":
                        if not regression:
                            prob += (torch.exp(z[0, y] - logsumexp(z))).detach().cpu().numpy()[0]
                            correct += torch.sum(pred == y).cpu().numpy()
                        l = criterion(z, y)
                        if stochastic:
                            loss = l
                        else:
                            loss += l
                        total += l.detach().cpu().numpy()
                        if regularize is not None:
                            r = regularize(classifier)
                            loss += r
                            reg += r.detach().cpu().numpy()

                    if dataset == "train" and (stochastic or start + 1 == len(X)):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    elif dataset == "test":
                        pred = pred.numpy()
                        if len(pred.shape) != 0:
                            pred = pred[0]
                        if state is not None:
                            pred = state[pred]

    #                     print(str(pred))
#                         logger.info(y + "\t" + str(pred))

                if dataset != "test" and n != 0:
                    res = {}
                    res["loss"] = total / float(n)
                    res["accuracy"] = correct / float(n)
                    res["soft"] = prob / float(n)
                    if any(map(math.isnan, y_score)):
                        res["auc"] = float("nan")
                        res["r2"] = float("nan")
                    elif regression:
                        res["auc"] = float("nan")
    #                     print(np.unique(y_score))
    #                     print(np.unique(y_true))

    #                     onehot_encoder = OneHotEncoder(sparse=False)
    #                     integer_encoded = np.reshape(y_score,(len(y_score), 1))
    #                     y_score = onehot_encoder.fit_transform(integer_encoded)
                        # invert first example
                        res["r2"] = sklearn.metrics.r2_score(y_true, y_score, multi_class='ovo')
                    else:
    #                     onehot_encoder = OneHotEncoder(sparse=False)
    #                     integer_encoded = np.reshape(y_score,(len(y_score), 1))
    #                     y_score = onehot_encoder.fit_transform(integer_encoded)
                        res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score, multi_class='ovo')
                        res["r2"] = float("nan")

#                     logger = logging.getLogger(__name__)
#                     logger.debug("        Loss           " + str(res["loss"]))
#                     logger.debug("        Accuracy:      " + str(res["accuracy"]))
#                     logger.debug("        Soft Accuracy: " + str(res["soft"]))
#                     logger.debug("        AUC:           " + str(res["auc"]))
#                     logger.debug("        R2:            " + str(res["r2"]))

#                     if regularize is not None:
#                         logger.debug("        Regularize:    " + str(reg / float(n)))
                if dataset == "train":
                    log.append([])
                log[-1].append((total / float(n), correct / float(n)) if n != 0 else (None, None))
                if dataset == "valid":
#                     print("-----------------------------------------------------------")
#                     print(res)                  
                    if res["loss"] <= best_res["loss"]:
                        best_res = res
                        best_model = copy.deepcopy(classifier.state_dict())
#                         if iterations - 1 ==0:
#                             with open(path+'y_score3_', 'wb') as fp:
#                                 pickle.dump(y_score, fp)
#                             pred= torch.stack(pred_)
#                             with open(path+'pred3_', 'wb') as fp:
#                                 pickle.dump(pred.tolist(), fp)  
#                             with open(path+'y_true3_', 'wb') as fp:
#                                 pickle.dump(y_true, fp)                              
    print("**********************************************")
    print(best_res)                        
    classifier.load_state_dict(best_model)
              

    # torch.save(classifier.state_dict(), "model.pt")
    return classifier, best_res


def logsumexp(x, dim=None):
    if dim is None:
        m = torch.max(x)
        return m + torch.log(torch.sum(torch.exp(x - m)))
    else:
        m, _ = torch.max(x, dim, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m), dim))

def train_class(Xtrain, centers=2,seed=0):
    gm = collections.defaultdict(list)
    count = collections.defaultdict(int)
    for X, y, *_ in Xtrain:
        gm[y].append(X)
        count[y] += 1
    
    for state in gm:
        gm[state] = np.concatenate(gm[state])
        model = sklearn.mixture.GaussianMixture(centers,random_state=seed)
        gm[state] = model.fit(gm[state])

    return (gm, count)

def eval_class(model, Xtest, path):
    gm, count = model
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    pred_=[]
    for X, y, *_ in Xtest:
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = sum(gm[state].score_samples(X))
            x = max(x, logp[state])
        y_score.append(logp[1] - logp[0])
        y_true.append(y)
        Z = 0
        for state in logp:
            logp[state] = math.exp(logp[state] - x) * count[state]
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state
        # total += math.log(logp[state])
        pred_.append(pred)
        correct += (pred == y)
        prob += logp[y]
    n = len(Xtest)

    res = {}
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score)

#     logger = logging.getLogger(__name__)
#     # logger.debug("        Generative Cross-entropy: " + str(total / float(n)))
#     logger.debug("        Generative Accuracy:      " + str(res["accuracy"]))
#     logger.debug("        Generative Soft Accuracy: " + str(res["soft"]))
#     logger.debug("        Generative AUC:           " + str(res["auc"]))  
    with open(path+'y_score_gmmclass', 'wb') as fp:
                            pickle.dump(y_score, fp)
    with open(path+'pred_gmmclass', 'wb') as fp:
                            pickle.dump(pred_, fp)  
    with open(path+'y_true_gmmclass', 'wb') as fp:
                            pickle.dump(y_true, fp)  
    return res



def train_patient(Xtrain, centers=2,seed=0):
    gm = collections.defaultdict(list)
    for (i, (X, y, *_)) in enumerate(Xtrain):
        model = sklearn.mixture.GaussianMixture(min(centers, X.shape[0]),random_state=seed)
#         print(len(X))
        model.fit(X)
        gm[y].append(model)
    
#         print("Train " + str(i + 1) + " / " + str(len(Xtrain)), end="\r")
        sys.stdout.flush()
#     print()
    return gm

def eval_patient(gm, Xtest,path):
    total = 0.
    correct = 0
    prob = 0.
    y_score = []
    y_true = []
    pred_=[]
    for (i, (X, y, *_)) in enumerate(Xtest):
        logp = {}
        x = -float("inf")
        for state in gm:
            logp[state] = list(map(lambda m: sum(m.score_samples(X)), gm[state]))
            x = max(x, max(logp[state]))
        # print(logp)
        Z = 0
        for state in logp:
            logp[state] = sum(map(lambda lp: math.exp(lp - x), logp[state]))
            Z += logp[state]
        pred = None
        for state in logp:
            logp[state] /= Z
            if pred is None or logp[state] > logp[pred]:
                pred = state
        total += -math.log(max(1e-50, logp[y]))
        pred_.append(pred)
        correct += (pred == y)
        prob += logp[y]
        y_score.append(logp[1])
        y_true.append(y)
    
#         print("Test " + str(i + 1) + " / " + str(len(Xtest)) + ": " + str(correct / float(i + 1)), end="\r", flush=True)
#     print()
    
    n = len(Xtest)
    res = {}
    res["ce"] = total / float(n)
    res["accuracy"] = correct / float(n)
    res["soft"] = prob / float(n)
    res["auc"] = sklearn.metrics.roc_auc_score(y_true, y_score)

#     logger = logging.getLogger(__name__)
#     logger.debug("        Genpat Cross-entropy: " + str(res["ce"]))
#     logger.debug("        Genpat Accuracy:      " + str(res["accuracy"]))
#     logger.debug("        Genpat Soft Accuracy: " + str(res["soft"]))
#     logger.debug("        Genpat AUC:           " + str(res["auc"]))
    with open(path+'y_score_gmmpat', 'wb') as fp:
                            pickle.dump(y_score, fp)
    with open(path+'pred_gmmpat', 'wb') as fp:
                            pickle.dump(pred_, fp)  
    with open(path+'y_true_gmmpat', 'wb') as fp:
                            pickle.dump(y_true, fp)    
    return res
