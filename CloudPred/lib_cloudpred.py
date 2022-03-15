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

def train(Xtrain, Xvalid, centers=2, regression=False, seed=0):
    X = np.concatenate([x for (x, *_) in Xtrain])
    y_label = [y for (x,y, *_) in Xtrain]
#________________________________________________________________________    
    X_train= Xtrain
    kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
    for elem in kf: 
                train_index, test_index=elem
                Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
                pass
#________________________________________________________________________    
 
    gm = []
    for X, *_ in Xtrain:
        gm.append(X)

    gm = np.concatenate(gm)
    gm = sklearn.mixture.GaussianMixture(n_components=centers, covariance_type="diag", random_state=seed).fit(gm)

    component = [Gaussian(torch.Tensor(gm.means_[i, :]),
                          torch.Tensor(1. / gm.covariances_[i, :])) for i in range(centers)]
    mixture = Mixture(component, gm.weights_)
    classifier = DensityClassifier(mixture, centers, len(np.unique(y_label)))

    X = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xtrain])
    if regression:
        y = torch.FloatTensor([y for (X, y, *_) in Xtrain])
    else:
        y = torch.LongTensor([y for (X, y, *_) in Xtrain])

    Xv = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xvalid])
    if regression:
        yv = torch.FloatTensor([y for (X, y, *_) in Xvalid])
    else:
        yv = torch.LongTensor([y for (X, y, *_) in Xvalid])

#     logger = logging.getLogger(__name__)
    # Set weights of classifier
    for lr in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
        optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9)
        if regression:
            criterion = torch.nn.modules.MSELoss()
        else:
            criterion = torch.nn.modules.CrossEntropyLoss()
        best_loss = float("inf")
        best_model = copy.deepcopy(classifier.pl.state_dict())
#         logger.debug("Learning rate: " + str(lr))
        for i in range(1000):
            z = classifier.pl(X)
            if regression:
                z = z[:, 1]
            loss = criterion(z, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            zv = classifier.pl(Xv)
            if regression:
                zv = zv[:, 1]
            loss = criterion(zv, yv)
#             if i % 100 == 0:
#                 logger.debug(str(loss.detach().numpy()))
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(classifier.pl.state_dict())
        classifier.pl.load_state_dict(best_model)

    reg = None
    #________________________________________________________________________    
#     Xtrain= X_train
#     Xvalid = []
    #________________________________________________________________________    

    return train_classifier(Xtrain, Xvalid, [], classifier, regularize=reg,
                                            iterations=1000, eta=1e-4, stochastic=True,
                                            regression=regression, seed=seed)


def eval(model, Xtest, regression=False, path=''):
    reg = None
    print('eval------------------------eval')
    model, res = train_classifier([], Xtest, [], model, regularize=reg,
                                                  iterations=1, eta=0, stochastic=True,
                                                  regression=regression,seed=0, path=path)
    return res


class Gaussian(torch.nn.Module):
    def __init__(self, mu, invvar):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.parameter.Parameter(mu)
        self.invvar = torch.nn.parameter.Parameter(invvar)

    def forward(self, x):
        invvar = torch.abs(self.invvar).clamp(1e-5)
        return -0.5 * (math.log(2 * math.pi) - torch.sum(torch.log(invvar))
                       + torch.sum((self.mu - x) ** 2 * invvar, dim=1))


class Mixture(torch.nn.Module):
    def __init__(self, component, weights):
        super(Mixture, self).__init__()
        self.component = torch.nn.ModuleList(component)
        self.weights = torch.nn.parameter.Parameter(torch.Tensor(weights).unsqueeze_(1))

    def forward(self, x):
        logp = torch.cat([c(x).unsqueeze(0) for c in self.component])
        shift, _ = torch.max(logp, 0)
        p = torch.exp(logp - shift) * self.weights
        return torch.mean(p / torch.sum(p, 0), 1)


class DensityClassifier(torch.nn.Module):
    def __init__(self, mixture, centers, states=2):
        super(DensityClassifier, self).__init__()
        self.mixture = mixture
        self.pl = PolynomialLayer(centers, states)

    def forward(self, x):
        self.d = self.mixture(x).unsqueeze_(0)
        return self.pl(self.d)


class PolynomialLayer(torch.nn.Module):
    def __init__(self, centers, states=2):
        super(PolynomialLayer, self).__init__()
        self.polynomial = torch.nn.ModuleList([Polynomial(centers) for _ in range(states - 1)])

    def forward(self, x):
        return torch.cat([torch.zeros(x.shape[0], 1)]
                         + [p(x).unsqueeze_(1) for p in self.polynomial], dim=1)


class Polynomial(torch.nn.Module):
    def __init__(self, centers=1, degree=2):
        super(Polynomial, self).__init__()
        self.centers = centers
        self.degree = degree
        self.a = torch.nn.parameter.Parameter(torch.zeros(degree, centers))
        self.c = torch.nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sum(sum([self.a[i, :] * (x ** (i + 1)) for i in range(self.degree)]), dim=1) + self.c

    def linear_reg(self, xy):
        x = np.concatenate(list(map(lambda x: x[0].reshape(1, -1), xy)))
        y = np.array(list(map(lambda x: x[1], xy)))
        y = 2 * y - 1
        x = np.concatenate([x ** (i + 1) for i in range(self.degree)] + [np.ones((x.shape[0], 1))], axis=1)
        w = np.dot(np.linalg.pinv(x), y)
        self.a.data = torch.Tensor(w[:-1].reshape(self.degree, self.centers))
        self.c.data = torch.Tensor([w[-1]])

def logsumexp(x, dim=None):
    if dim is None:
        m = torch.max(x)
        return m + torch.log(torch.sum(torch.exp(x - m)))
    else:
        m, _ = torch.max(x, dim, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m), dim))
    
def train_classifier(Xtrain, Xvalid, Xtest, classifier, regularize=None,
                     eta=1e-8, iterations=3, stochastic=True, cuda=False, state=None, regression=False,path='',seed=0):

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
#             kf = KFold(n_splits=5, shuffle= True, random_state=seed).split(X_train)
#         else:
#             kf = ['e']
#         for elem in kf: 
#             if len(Xtrain)>0:
#                 train_index, test_index=elem
#                 Xtrain, Xvalid = list(np.array(X_train)[train_index]), list(np.array(X_train)[test_index])
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
                    if iteration%100 ==0:
                        print("-----------------------------------------------------------")
                        print("Iteration #" + str(iteration + 1) + ":")
                        print(res)                                
                    if res["loss"] <= best_res["loss"]:
                        best_res = res
                        best_model = copy.deepcopy(classifier.state_dict())
                        if iterations - 1 ==0:
                            with open(path+'y_score', 'wb') as fp:
                                pickle.dump(y_score, fp)
                            pred= torch.stack(pred_)
                            with open(path+'pred', 'wb') as fp:
                                pickle.dump(pred.tolist(), fp)  
                            with open(path+'y_true', 'wb') as fp:
                                pickle.dump(y_true, fp)                              
    print("**********************************************")
    print(best_res)                        
    classifier.load_state_dict(best_model)
              

    # torch.save(classifier.state_dict(), "model.pt")
    return classifier, best_res

