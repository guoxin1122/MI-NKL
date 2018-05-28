#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:15:08 2018

@author: macgx
"""

import numpy as np
import random

np.random.seed(123) # control fitness matrix, not work for epistasis matrix
#def NK_fitness(x, K): # x is binary string vector
#    N = len(x)
#    # generate fitness matrix
#    fitnessMatrix = np.random.rand(N, 2**(K+1))
#    # initialize epistasis matrix where each element is the index of allele
#    epistasisMatrix = np.empty([N,K], dtype=int)
#    F_i = []
#    
#    for i in range(N):
#        # give valid values to epistasis matrix's i-th row, i.e. K random neighbours's indices of allele i
#        epistasisMatrix[i] = random.sample([j for j in range(N) if j != i ], k=K)
#        # initialize F_i_index using x_i 
#        F_i_ind = int(x[i])
#        # weighted sum of K neighbours's values of allele i
#        for j in range(K):
#            F_i_ind +=  2**j * int(x[epistasisMatrix[i][j]])
#        # get fitness component F_i, and append to list
#        F_i.append(fitnessMatrix[i][F_i_ind] )
#    f_value = sum(F_i) / N
#    
#    return f_value

#x = '101111001'
x_ = data[0].to_dict()

N = len(x_)
K = 2
nr = nz = nd = 5

Fitness = np.random.rand(N, 2**(K+1))

Epistasis = np.empty([N,K], dtype=int)
for i in range(N):
    Epistasis[i] = random.sample([j for j in range(N) if j != i ], k=K)

def normalize(x, lb, ub):        
    return (x-lb) / (ub - lb)    
for i in range(nr):
    x_['r_{}'.format(i)] = normalize(x_['r_{}'.format(i)], lb=-10, ub=10)
    x_['z_{}'.format(i)] = normalize(x_['z_{}'.format(i)], lb=0, ub=19)
    
x = list(x_.values())

F = []
for i in range(N):
    ai = [] 
    ax = []
    if isinstance(x[i], float): # real and integer number
        for j in range(2**(K+1)):
            if j == 0:
                ai.append(Fitness[i][j])
                ax.append(Fitness[i][j])
            else:
                # generate ai[j]
                a_l_sum = 0
                l_and_j = []
                for l in range(j):
                    if l&j not in l_and_j:
                        a_l_sum += ai[l&j]
                        l_and_j.append(l&j)
                ai.append(Fitness[i][j] - a_l_sum)
                # compute ai[j] * x_i * x_i_neighbours
                x_ik_product = 1
                for k in range(K):
                    x_ik_product *= x[Epistasis[i][k]] ** ((2**(k+1) & j) / 2**(k+1))
                ax.append(ai[j] * x[i] ** (1&j) * x_ik_product)
        F.append(sum(ax))
    elif isinstance(x[i], int): # nominal discrete number
        for j in range(2**(K+1)):
            Fitness[i][j]
#        F_i_ind = x[i]
#        for k in range(K):
#            F_i_ind +=  2**(k+1) * x[Epistasis[i][k]]
        print(F_i_ind)
#        F.append(Fitness[i][F_i_ind]) 
#f_value = sum(F) / N





##def Fitness_i(xi, K):
#N_r = 3
#K_r = 2
#x_r_epistasisMat = np.empty([N_r,K_r], dtype=int)
#for i in range(N_r):
#    x_r_epistasisMat[i] = random.sample([j for j in range(N_r) if j != i ], k=K_r)
## TODO: x is real number and change int(x[epistasisMatrix[0][k]]) to float
## generate a_i 1d matrix
#
#x_r = np.random.rand(N_r)
#a_i = []  
#ax_item = [] 
#F_r = []
#for i in range(N_r):
#    # compute F_i
#    for j in range(2**(K+1)):
#        if j == 0:
#            a_i.append(fitnessMatrix[i][0])
#            ax_item.append(a_i[0])
#        else:
#            # generate a_i[j]
#            a_l_sum = 0
#            l_and_j = []
#            for l in range(j):
#                if l&j not in l_and_j:
#                    a_l_sum += a_i[l&j]
#            a_i.append(fitnessMatrix[i][j] - a_l_sum)
#            # compute a_i[j] * x_i
#            x_ik = 1
#            for k in range(K):
#                x_ik *= x_r[x_r_epistasisMat[i][k]] ** ((2**(k+1) & j) / 2**(k+1))
#            ax = a_i[j] * x_r[i] ** (1&j) * x_ik
#            ax_item.append(ax)
#    
#    F_i = sum(ax_item)
#    F_r.append(F_i)
##    print(F_i)
#
##for k in range(K):
##    x_ik *= int(x[epistasisMatrix[i][k]]) ** (2**k & j)
##ax = a_i[j] * xi ** (1&j) * x_ik
#    
#    
#for i in range(N):
#    # give valid values of one row to epistasis matrix, i.e. K random neighbours's indices fo allele i
#    epistasisMatrix[i] = random.sample([j for j in range(N) if j != i ], k=K)
#    # initialize F_i_index using x_i itself
#    F_i_ind = int(x[i])
#    # weighted sum of K neighbours's values of allele i
#    for j in range(K):
#        F_i_ind +=  2**j * int(x[epistasisMatrix[i][j]])
#    F_.append(fitnessMatrix[i][F_i_ind] )
#f_value = sum(F_) / N

































        

