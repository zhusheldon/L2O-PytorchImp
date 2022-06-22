# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:12:08 2022

@author: sheld
"""
from abc import ABC, abstractmethod
from better_abc import ABCMeta, abstract_attribute

import sys
import os
import numpy as np
from time import time

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import scipy
from scipy.stats import ortho_group

import copy

import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self, data:torch.Tensor):
        super().__init__()
        self.x = torch.nn.Parameter(data)

class Fcn(ABC):
    @abstract_attribute
    def num_dim(self):
        pass
    
    @abstract_attribute
    def optimal_val(self):
        pass
    
    @abstract_attribute
    def optimal_x(self):
        pass
    
    @abstractmethod
    def objectiveFcn(self, var):
        pass
        
        

class QuadraticFcn(Fcn):
    def __init__(self, num_dim, **kwargs):
        eig_vecs = torch.tensor(
            ortho_group.rvs(dim=(num_dim)), dtype=torch.float
        )
        eig_vals = torch.rand(num_dim) * 80 + 1
        self.A = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
        self.b = torch.normal(0, 1 / np.sqrt(num_dim), size=(num_dim,))
        
        self.optimal_x = torch.tensor(scipy.linalg.solve(self.A.numpy(), -self.b.numpy(), assume_a="pos"))
        self.optimal_val = self.objectiveFcn(Model(self.optimal_x)).item()
        self.x0 = Model(torch.normal(np.random.rand()*10, 0.5 / np.sqrt(num_dim), size=(num_dim,)))
    
    def objectiveFcn(self, var):
        x = var.x
        return 0.5 * x.T @ self.A @ x + self.b.T @ x
    
    
class RosenbrockFcn(Fcn):
    def __init__(self, num_dim):
        self.optimal_x = torch.ones(num_dim)
        self.optimal_val=self.objectiveFcn(Model(self.optimal_x)).item()
        self.x0 = Model(torch.normal(np.random.rand()*10, 0.5 / np.sqrt(num_dim), size=(num_dim,)))

        # Below are used for create generalized Rosenbrock functions
        # self.a = 1 
        # self.b=100
    
    def objectiveFcn(self, var):
        x = var.x
        return torch.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0+(1-x[:-1])**2.0)
    
# class LogisticRegressionLossFcn(Fcn):
#     def __init__(self, num_dim):
#         self.

# class customQuadFcnDataset(Dataset):
#     def __init__(self,num_samples):
#         self.quadFcns = [QuadraticFcn(2) for _ in range(num_samples)]
    
#     def __len__(self,):
#         return len(self.quadFcns)
    
#     def __getitem__(self,idx):
#         return self.quadFcns[idx], self.quadFcns[idx].optimal_x

# QuadraticFcnSamples=list()
# for i in range(1000):
#     QuadraticFcnSamples.append(QuadraticFcn(2))
    
# RosenbrockFcnSamples=list()
# for i in range(1000):
#     RosenbrockFcnSamples.append(RosenbrockFcn(2))

# QuadDataLoader = DataLoader(customQuadFcnDataset(100), batch_size= 10, shuffle=True)
# train_features, train_labels = next(iter(QuadDataLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")


np.random.seed(0)
torch.manual_seed(0)
sample = RosenbrockFcn(2)
def optimizeObjFcn(sample, optimizerAlg, **kwargs):
    model = copy.deepcopy(sample.x0)
    
    obj_function = sample.objectiveFcn

    optimizer = optimizerAlg(model.parameters(), **kwargs)
    values = []
    trajectory = []
    
    def closure():
        trajectory.append(copy.deepcopy(model))
        optimizer.zero_grad()

        obj_value = obj_function(model)
        obj_value.backward()
        
        values.append(obj_value.item())
        return obj_value

    for i in range(200):
        optimizer.step(closure)
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            values = values[:-1]
            break
    return np.nan_to_num(values), trajectory
    
valsAdagrad, trajAdagrad = optimizeObjFcn(sample, torch.optim.Adagrad, lr=1)
# valsSgd, trajSgd = optimizeObjFcn(sample, torch.optim.SGD, lr=1e0)
valsRMSProp, trajRMSProp = optimizeObjFcn(sample, torch.optim.RMSprop, lr=1e-2)
valsAdam, trajAdam = optimizeObjFcn(sample, torch.optim.Adam, lr=1e-1)
valsLbfgs, trajLbfgs = optimizeObjFcn(sample, torch.optim.LBFGS, lr=1, max_iter=1)

print(f"Adagrad best loss: {valsAdagrad.min()}")
# print(f"SGD best loss: {valsSgd.min()}")
print(f"RMSProp best loss: {valsRMSProp.min()}")
print(f"Adam best loss: {valsAdam.min()}")
print(f"L-BFGS best loss: {valsLbfgs.min()}")


plt.figure(figsize=(6,4), dpi=150)

plt.title('Convex Quadratic objective value')

plt.plot(valsAdagrad-sample.optimal_val, label='Adagrad')
# plt.plot(valsSgd-sample.optimal_val, label='SGD')
plt.plot(valsRMSProp-sample.optimal_val, label='RMSprop')
plt.plot(valsAdam-sample.optimal_val, label='Adam')
plt.plot(valsLbfgs-sample.optimal_val, label='L-BFGS')

plt.xlabel('Iterations')
plt.legend()
plt.show()


def plot_trajectories(trajectories, problem, get_weights, set_weights):
    """Plot optimization trajectories on top of a contour plot.

    Parameters:
        trajectories (List(nn.Module))
        problem (dict)
        get_weights (Callable[[], Tuple[float, float]])
        set_weights (Callable[[float, float], None])

    """
    data = {}
    for name, traj in trajectories.items():
        data[name] = np.array([get_weights(model) for model in traj])

    xmin = min(np.array(d)[:, 0].min() for d in data.values())
    ymin = min(np.array(d)[:, 1].min() for d in data.values())
    xmax = max(np.array(d)[:, 0].max() for d in data.values())
    ymax = max(np.array(d)[:, 1].max() for d in data.values())

    X = np.linspace(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
    Y = np.linspace(ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2)

    model = copy.deepcopy(problem.x0)
    Z = np.empty((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            set_weights(model, X[i], Y[j])
            Z[j, i] = problem.objectiveFcn(model)

    plt.figure(figsize=(10, 6), dpi=500)
    plt.contourf(X, Y, Z, 30, cmap="RdGy")
    plt.colorbar()

    for name, traj in data.items():
        plt.plot(traj[:, 0], traj[:, 1], label=name)

    plt.title("Convex Quadratic Trajectory Plot")
    plt.plot(*get_weights(problem.x0), "bo")
    plt.legend()

    plt.plot()
    plt.show()
    

def get_weights(model):
    return model.x[0].item(), model.x[1].item()

def set_weights(model, w1, w2):
    with torch.no_grad():
        model.x[0] = w1
        model.x[1] = w2
    
plot_trajectories({'Adam': trajAdam,
                   'RMSProp': trajRMSProp}, sample, get_weights, set_weights)