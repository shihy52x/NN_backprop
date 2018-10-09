#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#  Copyright © XYM
# CreateTime: 2018-10-08 15:27:09

import numpy as np
import pdb

def init_w(layer_nums_list):
    """
    initialze weights. the first column is the bias unit
    """
    weights = []
    for i in range(0, len(layer_nums_list) -1 ):
        l_in = layer_nums_list[i]
        l_out = layer_nums_list[i+1]
        weights.append(np.random.rand(l_out, l_in))
    return weights

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def sigmoid_grad(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g

def RELU(z):
    x = z.reshape(-1)
    g = np.zeros(len(x))
    for i in range(0,len(x)):
        if x[i] > 0:
            g[i] = x[i]
    return g.reshape(-1,1)

def RELU_grad(z):
    x = z.reshape(-1)
    g = np.piecewise(x,[x<=0, x>0],[0,1])
    return g.reshape(-1,1)
 
def fw_prop(weights, x):
    # Forward propogation. 
    # Use RELU for hidden layer
    # Use sigmoid for hidden layer
    a = x
    z_cach = []
    a_cach = []
    for i in range(0, len(weights)-1):
        W = weights[i]
        z = np.dot(W,a)
        z_cach.append(z)
        a = RELU(z)
        a_cach.append(a)
    W = weights[-1]
    z = np.dot(W,a)
    z_cach.append(z)
    a = sigmoid(z)
    a_cach.append(a)
    return a, z_cach, a_cach

def bw_prop(weights, z_cach, a_cach):
    # Backward propoagation

    dz = 1
    for i in range(len(weights)-1, 0,-1):
        a_prev = a_cach[i - 1]
        z_prev = z_cach[i - 1]
        W    = weights[i]
        da_prev = np.dot(W.T, dz)
        dz_prev = da_prev * RELU_grad(z_prev)
        da = da_prev
        dz = dz_prev
    W = weights[0]
    dx = np.dot(W.T, dz)
    return dx
     



# Using three layers network.
x = np.array([[1.0],[2.0]])
T = 0.6
L0 = len(x)
L1 = 2
L3 = 1


np.random.seed(1234)
weights = init_w([L0,L1,L3])
a_pred, z_cach, a_cach = fw_prop(weights, x)
print "x =", x
print "T = ", T,' a_pred = ', a_pred

z_out_pred = z_cach[-1]
z_out_target = -np.log(1.0/T -1)
dz_out = z_out_target - z_out_pred


dx = bw_prop(weights, z_cach, a_cach)
deltx = dz_out/dx[0]
x[0] = x[0] + deltx


print "after correction"
print "x= ", x

a_pred, z_cach, a_cach = fw_prop(weights, x)
print "T = ",T,' a_pred = ', a_pred
