#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:15:08 2018

@author: macgx
"""

import numpy as np
import random

def MINKL(x_):
    
    np.random.seed(123) # control fitness matrix, not work for epistasis matrix
    N = len(x_)
    K = 3
    nr = nz = nd = 5
    
    if K > (N-1):
        raise ValueError("K can not be larger than (N-1)")
    
    Fitness = np.random.rand(N, 2**(K+1))
    
    Epistasis = np.empty([N,K], dtype=int)
    for i in range(N):
        Epistasis[i] = random.sample([j for j in range(N) if j != i ], k=K)
    
    def normalize(x, lb, ub):        
        return (x-lb) / (ub - lb)    
    for i in range(nr):
        x_['r_{}'.format(i)] = normalize(x_['r_{}'.format(i)], lb=-10, ub=10)
        x_['z_{}'.format(i)] = normalize(x_['z_{}'.format(i)], lb=0, ub=19)
        
    def bitlist_to_integer(bitlist):
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        return out
        
    x = list(x_.values())
    
    F = []
    for i in range(N):
        # check the variable type of x[i]
        if isinstance(x[i], float): # real and integer type
            D = 0  # D nominal discrete variables interact with a continuous or integer variable
            x_d = []
            x_rz = []
            # check the variable type of x[i]'s neighbour(s)
            for k in range(K): # count nominal discrete variables neighbours, and store x[i]'s two types of neighbours respectively
                if isinstance(x[Epistasis[i][k]], int):
                    x_d.append(x[Epistasis[i][k]])
                    D += 1
                elif isinstance(x[Epistasis[i][k]], float): 
                    x_rz.append(x[Epistasis[i][k]])
                else:
                    raise ValueError("Variable type of x[i]'s neighbour is not int or float")
                    
            if D > 0: # at least one of x[i]'s neighbours is nominal discrete type
                assert len(x_d) > 0
                K_new = K - D
                Fitness_segment = np.split(Fitness[i], 2 ** len(x_d))
                Fitness_new = Fitness_segment[bitlist_to_integer(x_d)]
                ai = [] 
                ax = []
                for j in range(2 ** (K_new+1)):
                    if j == 0:
                        ai.append(Fitness_new[j])
                        ax.append(Fitness_new[j])
                    else:
                        # generate ai[j]
                        a_l_sum = 0
                        l_and_j = []
                        for l in range(j):
                            if l&j not in l_and_j:
                                a_l_sum += ai[l&j]
                                l_and_j.append(l&j)
                        ai.append(Fitness_new[j] - a_l_sum)
                        # compute ai[j] * x_i * x_i_neighbours
                        x_ik_product = 1
                        for k in range(K_new):
                            x_ik_product *= x_rz[k] ** ((2**(k+1) & j) / 2**(k+1)) # here use x_rz 
                        ax.append(ai[j] * x[i] ** (1&j) * x_ik_product)
                F.append(sum(ax))
                    
            else: # all of x[i]'s neighbours are real or integer type
                assert D == 0
                ai = [] 
                ax = []
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
                            x_ik_product *= x[Epistasis[i][k]] ** ((2**(k+1) & j) / 2**(k+1)) # here use Epistasis
                        ax.append(ai[j] * x[i] ** (1&j) * x_ik_product)
                F.append(sum(ax)) 
    
        elif isinstance(x[i], int): # nominal discrete type
            F_i_ind = x[i]
            for k in range(K):
                if isinstance(x[Epistasis[i][k]], int): 
                    F_i_ind +=  2**(k+1) * x[Epistasis[i][k]]
                elif isinstance(x[Epistasis[i][k]], float):# if x[i]'s neighbour is not nominal discrete type       
                    F_i_ind +=  2**(k+1) * round(x[Epistasis[i][k]]) # round it to integer {0,1}
                else:
                    raise ValueError("Can not deal with nominal discrete variable")
            F.append(Fitness[i][F_i_ind]) 
        else:
            raise ValueError("Variable type is not int or float")
    f_value = sum(F) / N
#    print(f_value)
    return f_value

x_ = data[4].to_dict()
print(MINKL(x_))

# =============================================================================
# # NK fitness model
# 
# =============================================================================
#x = '101111001'
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
#    return f_value


