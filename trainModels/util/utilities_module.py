#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Originally Adapted from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py
Then adapted from: https://github.com/nickhnelsen/FourierNeuralMappings 

"""

import torch
import operator
from functools import reduce
import numpy as np
import scipy.io
# import hdf5storage
import pdb

#################################################
#
# utilities
#
#################################################

def to_torch(x, to_float=True):
    """
    send input numpy array to single precision torch tensor
    """
    if to_float:
        if np.iscomplexobj(x):
            x = x.astype(np.complex64)
        else:
            x = x.astype(np.float32)
    return torch.from_numpy(x)


# def validate(f, fhat):
#     '''
#     Helper function to compute relative L^2 error of approximations.
#     Takes care of different array shape interpretations in numpy.

#     INPUTS:
#             f : array of high-fidelity function values
#          fhat : array of approximation values

#     OUTPUTS:
#         error : float, relative error
#     '''
#     f, fhat = np.asarray(f).flatten(), np.asarray(fhat).flatten()
#     return np.linalg.norm(f-fhat) / np.linalg.norm(f)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    # Reference: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})


# class MatReader(object):
#     """
#     reading data
#     """
#     def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True,
#                  variable_names=None):
#         super(MatReader, self).__init__()

#         self.file_path = file_path
#         self.to_torch = to_torch
#         self.to_cuda = to_cuda
#         self.to_float = to_float
#         self.variable_names = variable_names    # a list of strings (key values in mat file)

#         self.data = None
#         self.old_mat = None
#         self._load_file()

#     def _load_file(self):
#         try:
#             self.data = scipy.io.loadmat(self.file_path, variable_names=self.variable_names)
#             self.old_mat = True
#         except:
#             self.data = hdf5storage.loadmat(self.file_path, variable_names=self.variable_names)
#             self.old_mat = False

#     def load_file(self, file_path):
#         self.file_path = file_path
#         self._load_file()

#     def read_field(self, field):
#         x = self.data[field]
#         if self.to_float:
#             if np.iscomplexobj(x):
#                 x = x.astype(np.complex64)
#             else:
#                 x = x.astype(np.float32)
#         if self.to_torch:
#             x = torch.from_numpy(x)
#             if self.to_cuda:
#                 x = x.cuda()
#         return x

#     def set_cuda(self, to_cuda):
#         self.to_cuda = to_cuda

#     def set_torch(self, to_torch):
#         self.to_torch = to_torch

#     def set_float(self, to_float):
#         self.to_float = to_float


# class UnitGaussianNormalizer(object):
#     """
#     normalization, pointwise gaussian
#     """
#     def __init__(self, x, eps=1e-6):
#         super(UnitGaussianNormalizer, self).__init__()

#         # x has sample/batch size as first dimension (could be ntrain*n or ntrain*T*n or ntrain*n*T)
#         self.mean = torch.mean(x, 0)
#         self.std = torch.std(x, 0)
#         self.eps = eps

#     def encode(self, x):
#         x = (x - self.mean) / (self.std + self.eps)
#         return x

#     def decode(self, x, sample_idx=None):
#         if sample_idx is None:
#             std = self.std + self.eps # n
#             mean = self.mean
#         else:
#             if len(self.mean.shape) == len(sample_idx[0].shape):
#                 std = self.std[sample_idx] + self.eps  # batch*n
#                 mean = self.mean[sample_idx]
#             if len(self.mean.shape) > len(sample_idx[0].shape):
#                 std = self.std[:,sample_idx]+ self.eps # T*batch*n
#                 mean = self.mean[:,sample_idx]

#         # x is in shape of batch*n or T*batch*n
#         x = (x * std) + mean
#         return x

#     def cuda(self):
#         self.mean = self.mean.cuda()
#         self.std = self.std.cuda()

#     def cpu(self):
#         self.mean = self.mean.cpu()
#         self.std = self.std.cpu()
class H1Loss(object):
    """
    loss function with rel/abs H1 loss
    """
    def __init__(self, d=2, size_average=True, reduction=True, eps=1e-6):
        # super(LpLoss, self).__init__()

        self.d = d
        self.p = 2
        self.reduction = reduction
        self.size_average = size_average
        self.eps =eps
    
    def rel_H1(self, x, y):
        num_examples = x.size()[0]
        grid_edge = x.size()[-1]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        h = 1.0 / (grid_edge - 1.0)
        x_grad = torch.gradient(x,dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1) # num_examples, (grad component) 2, (\chi_1, \chi_2) 2, grid_edge, grid_edge
        y_grad = torch.gradient(y, dim = (-2,-1), spacing = h)
        y_grad1 = y_grad[0].unsqueeze(1) # Component 1 of the gradient
        y_grad2 = y_grad[1].unsqueeze(1) # Component 2 of the gradient
        y_grad = torch.cat((y_grad1, y_grad2), 1) 

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        diff_grad_norms = torch.norm(x_grad.reshape(num_examples, -1) - y_grad.reshape(num_examples, -1), self.p, 1)
        diff_norms = (diff_norms**2 + diff_grad_norms**2)**(1/2)
        y_norms = ((torch.norm(y.reshape(num_examples,-1), self.p, 1))**2 + (torch.norm(y_grad.reshape(num_examples,-1), self.p, 1))**2)**(1/2)
        y_norms += self.eps     # prevent divide by zero
        rel_norms = torch.div(diff_norms,y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(rel_norms)
            else:
                return torch.median(rel_norms)

        return rel_norms

    def squared_H1(self, x, y):
        num_examples = x.size()[0]
        grid_edge = x.size()[-1]
        x = torch.squeeze(x) # shape is num_examples x channels_out x  grid_edge x grid_edge
        y = torch.squeeze(y) # shape is num_examples x channels_out x grid_edge x grid_edge 
        h = 1.0 / (grid_edge - 1.0)

        x_grad = torch.gradient(x, dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1) # num_examples, (grad component) 2, (\chi_1, \chi_2) 2, grid_edge, grid_edge
        y_grad = torch.gradient(y, dim = (-2,-1), spacing = h)
        y_grad1 = y_grad[0].unsqueeze(1) # Component 1 of the gradient
        y_grad2 = y_grad[1].unsqueeze(1) # Component 2 of the gradient
        y_grad = torch.cat((y_grad1, y_grad2), 1) # num_examples, 2, grid_edge, grid_edge

        diff_L2 = h*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, dim = 1)

        grad_euclidean = torch.norm(x_grad - y_grad, 2, dim = 1)
        diff_grad_L2 = h*torch.norm(grad_euclidean.reshape(num_examples,-1),2,1)
        sum_sq = diff_L2**2 + diff_grad_L2**2

        if self.reduction:
            if self.size_average:
                return torch.mean(sum_sq)
            else:
                return torch.sum(sum_sq)

        return sum_sq


class LpLoss(object):
    """
    loss function with rel/abs Lp norm loss
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True, eps=1e-6):
        super(LpLoss, self).__init__()

        if not (d > 0 and p > 0):
            raise ValueError("Dimension d and Lp-norm type p must be postive.")

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps =eps

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[-1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        pdb.set_trace()
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        y_norms += self.eps     # prevent divide by zero
        mean_y_norm = torch.mean(y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/mean_y_norm)
            else:
                return torch.sum(diff_norms/mean_y_norm)

        return diff_norms/mean_y_norm

    def __call__(self, x, y):
        return self.squared_H1(x, y)


class Sobolev_Loss(object):
    '''
    Loss object to compute H_1 loss or W_{1,p} loss, relative or absolute
    Assumes input shape is (num_examples, channels_out, grid_edge, grid_edge)
    Returns array of shape (num_examples,)
    '''
    def __init__(self, d=2, p=2, eps = 1e-6):
        self.d = d
        self.p = p
        self.eps =eps
    
    def compute_grad(self,x):
        grid_edge = x.size()[-1]
        h = 1.0 / (grid_edge - 1.0)

        x_grad = torch.gradient(x, dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1)
        return x_grad
    
    def Lp_norm(self,x):
        num_examples = x.size()[0]
        return torch.norm(x.reshape(num_examples,-1), self.p, dim=1)
    
    def Lp_err(self,x,y):
        return self.Lp_norm(x-y)
    
    def Lp_rel_err(self,x,y):
        return self.Lp_err(x,y)/(self.Lp_norm(y) + self.eps)
    
    def W1p_norm(self,x):
        x_grad = self.compute_grad(x)
        return (self.Lp_norm(x)**self.p + self.Lp_norm(x_grad)**self.p)**(1/self.p)
    
    def W1p_err(self,x,y):
        return self.W1p_norm(x-y)
    
    def W1p_rel_err(self,x,y):
        return self.W1p_err(x,y)/(self.W1p_norm(y)+self.eps)
    
def count_params(model):
    """
    print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def compute_Abar(A, chi):
    '''
    Computes Abar from A and chi
    Abar = \int_{\Td} (A + A\grad\chi^T) dx
    chi has shape (num_examples, channels_out, grid_edge, grid_edge)
    A has shape (num_examples, 3, grid_edge, grid_edge)
    '''
    '''
    num_examples = chi.size()[0]
    grid_edge = chi.size()[-1]
    h = 1.0 / (grid_edge - 1.0)
    
    # Add off-diagonal entry back to A
    off_diag = A[:,2,:,:].unsqueeze(1)
    A = torch.cat((A[:,:2,:,:],off_diag,A[:,:2,:,:]),1)
    
    # Compute grad chi
    chi_grad = torch.gradient(chi, dim = (-2,-1), spacing = h)
    chi_grad1 = chi_grad[0].unsqueeze(1) # Component 1 of the gradient
    chi_grad2 = chi_grad[1].unsqueeze(1) # Component 2 of the gradient
    chi_grad = torch.cat((x_grad1, x_grad2), 1)

    # Compute integrand
    integrand = [[A[n,:,:,:]]]







def loss_report(y_hat, y_true, A_true, model_name):
    H1_loss_func = Sobolev_Loss(p = 2)
    W1_10_loss_func = Sobolev_Loss(p = 10)

    H1_losses = H1_loss_func.W1p_err(y_hat,y_true)
    H1_rel_losses = H1_loss_func.W1p_rel_err(y_hat,y_true)

    W1_10_losses = W1_10_loss_func.W1p_err(y_hat,y_true)
    W1_10_rel_losses = W1_10_loss_func.W1p_rel_err(y_hat,y_true)

    H1_mean = torch.mean(H1_losses)
    H1_med = torch.median(H1_losses)
    H1_rel_mean = torch.mean(H1_rel_losses)
    H1_rel_med = torch.median(H1_rel_losses)

    W1_10_mean = torch.mean(W1_10_losses)
    W1_10_med = torch.median(W1_10_losses)
    W1_10_rel_mean = torch.mean(W1_10_rel_losses)
    W1_10_rel_med = torch.median(W1_10_rel_losses)

