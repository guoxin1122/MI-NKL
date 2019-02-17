#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import datetime
import math
import logging
import csv
from collections import OrderedDict
import random

class MINKL(object):
    def __init__(self, rng=None, K=None, bestFitness=999, Nr=5, Nz=5, Nd=5, L=2,
                 logfile=None, dim=None, optimizer=None, setConsoleHandler=False, logVerbose=False): # dim is assigned [15,20,25]
        self.rng = rng
        self.K = K
        self.bestFitness = bestFitness
        self.optimizer = optimizer
        
        self.Nr = Nr
        self.Nz = Nz
        self.Nd = Nd
        self.L = L
        
        self.N = self.Nr + self.Nz + self.Nd

        ## in case of run various dim:
        # self.dim = dim
        # self.Nr = self.dim - self.Nz - self.Nd
        # self.N = self.dim

        self.logfile = logfile
        self.logVerbose = logVerbose
        self.setConsoleHandler = setConsoleHandler

        # self._get_logger(self.logfile, self.setConsoleHandler)

        np.random.seed(self.rng) # control fitness matrix
        random.seed(self.rng) # control epistasis matrix

        self.typeOf = self.initTypeOf()
        self.E = self.initE()
        self.F = self.initF()

    def __call__(self, x_, *args, **kwargs): # x_ is given as an array by fgeneric
        #### preprocess x_ in order to adapte different optimizors
        def normalize(x, lb, ub):
            return (x-lb) / (ub - lb)
        # print("1. x_ = {}".format(x_))
        if self.optimizer == 'SMAC':
            # in smac, convert dictionary input to list
            # Note 1: in mithril server, dictionary x_ is not ordered !
            # Note 2: smac convert all of categorical hyperparameters to str type, so need to change to int
            from smac.configspace import Configuration
            assert isinstance(x_, Configuration)
            x_dict = x_._values 
            od = OrderedDict(sorted(x_dict.items())) # {'d_0':.., 'd_1':.., ..., 'r_0':.., 'r_1':.., ..., 'z_0':.., 'z_1':.., ...}
            x_ = list(od.values())
            x_d_str = x_[:self.Nd]
            x_d = [int(i) for i in x_d_str]
            x_r = [normalize(x_[i], -10, 10) for i in range(self.Nd, self.Nd+self.Nr)]
            x_z = [normalize(x_[i], 0, 19) for i in range(self.Nd+self.Nr, self.N)]
        elif self.optimizer == 'mipego':
            assert isinstance(x_, dict)
            # Note: in mithril server, dictionary x_ is not ordered !
            od = OrderedDict(sorted(x_.items())) # apply OrderedDict(sorted()), then od is ordered as 
                                                # {'d_0':.., 'd_1':.., ..., 'r_0':.., 'r_1':.., ..., 'z_0':.., 'z_1':.., ...}
            x_ = list(od.values())
            x_d = x_[:self.Nd]
            x_r = [normalize(x_[i], -10, 10) for i in range(self.Nd, self.Nd+self.Nr)]
            x_z = [normalize(x_[i], 0, 19) for i in range(self.Nd+self.Nr, self.N)]

        self.individual = x_r + x_z + x_d
        # self.individual = x_r + x_z + x_d
        self.fitness = self.evaluate(self.individual) # MINKL instance self.evaluate() need a list as input

        return self.fitness

    def _get_logger(self, logfile, setConsoleHandler):
        """
        When logfile is None, no records are written
        """
        # Multiple calls to logging.getLogger('someLogger') return a reference to the same logger object. 
        # This is true not only within the same module, 
        # but also across modules as long as it is in the same Python interpreter process

        self.logger = logging.getLogger(self.__class__.__name__) # self.__class__.__name__ is MINKL
        self.logger.setLevel(logging.WARNING) #TODO: what is original level?
        formatter = logging.Formatter('- %(asctime)s [%(levelname)s] -- '
                                      '[- %(process)d - %(name)s] %(message)s')

        # create console handler and set level to debug
        if setConsoleHandler:
            #define a Handler which writes INFO messages or higher to the sys.stderr
            ch = logging.StreamHandler() 
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # create file handler and set level to debug
        if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def initE(self):
        E = np.empty([self.N, self.K], dtype=int)
        for i in range(self.N):
            E[i] = random.sample([j for j in range(self.N) if j != i ], k=self.K)
        return E

    def initF(self):
        F = [9] * self.N
        for i in range(self.N):
            num_ordinals = self.K + 1
            num_nominals = 0
            if self.typeOf[i] == self.variableType['NOMINAL']:
                num_nominals = 1
            for k in range(self.K):
                neighbour = self.E[i][k]
                if self.typeOf[neighbour] == self.variableType['NOMINAL']:
                   num_nominals = num_nominals + 1
            num_ordinals = num_ordinals - num_nominals
            maxNominals = int(math.pow(self.L, num_nominals))
            maxOrdinals = int(math.pow(2, num_ordinals))
            F[i] = [8] * maxNominals
            for norminal in range(maxNominals):
                F[i][norminal] = [7] * maxOrdinals
                for ordinal in range(maxOrdinals):
                    F[i][norminal][ordinal] = np.random.uniform(0.00001, 1.0) # exclusive 0 and 1
        return F

    def initTypeOf(self):
        self.variableType = {'REAL':0, 'INTEGER':1, 'NOMINAL':2}
        typeOf = [self.variableType['NOMINAL']] * self.N
        for i in range(self.Nr):
           typeOf[self.Nd+i] = self.variableType['REAL']
        for i in range(self.Nz):
            typeOf[self.Nd+self.Nr+i] = self.variableType['INTEGER']
        #TODO: import enum34, convert self.variableType to enumeration type
        ## the following code has pickle problem during parallel
        # def _enum(**enums):
        #     return type('Enum', (), enums)
        # self.variableType = _enum(REAL=0, INTEGER=1, NOMINAL=2)
        # typeOf = [self.variableType.NOMINAL] * self.N
        # for i in range(self.Nr):
        #    typeOf[self.Nd+i] = self.variableType.REAL
        # for i in range(self.Nz):
        #     typeOf[self.Nd+self.Nr+i] = self.variableType.INTEGER
        return typeOf

    ########################## start computing the fitness

    # construct A
    def constructA(self, hypercubeCornerFitnessValues): # do not input A
        corners = len(hypercubeCornerFitnessValues)
        A = [hypercubeCornerFitnessValues[0]] * corners
        for corner in range(1, corners):
            A[corner] = hypercubeCornerFitnessValues[corner]-A[0]
            for j in range(1, corner):
                if j == (corner & j):
                    A[corner] -= A[j]
        return A

    def nthBit(self, a, n):
        return ((a>>(n))%2)==1

    def power(self, x, y):
        if x == 0:
            return 0
        if y == 0:
            return 1
        result = x
        for i in range(1, y):
            result = int(result * x)
        return result

    # implement the equation 3 in the way of equation 3.1
    def f(self, A, X):
        aSize = len(A)
        xSize = len(X)
        assert aSize == int(math.pow(2,xSize))
        result = A[0]
        term_result = 0
        for a in range(1, aSize):
            term_result = A[a]
            for n in range(xSize):
                if self.nthBit(a, n):
                    term_result *=X[n]
            result+=term_result
        return result

    # compute fitness of gene
    def fitnessOfGene(self, individual, gene):# we assume individual is list
        nominalIndex = 0
        nominals = 0
        ordinals = 0
        multiplier = 0

        X_ordinal = []

        ## get nominalIndex
        # check gene itself's type
        if self.typeOf[gene] == self.variableType['NOMINAL']:
            nominals += 1
            nominalIndex = int(individual[gene])

        else:
            ordinals += 1
            X_ordinal.append(individual[gene])
        # check neighbour's type
        for k in range(self.K):
            neighbour = self.E[gene][k]
            if self.typeOf[neighbour] == self.variableType['NOMINAL']:
                multiplier = int(self.power(self.L, nominals))
                nominalIndex += multiplier * int(individual[neighbour])
                nominals += 1
            else:
                X_ordinal.append(individual[neighbour])
                ordinals += 1


        result = 0
        assert (self.K+1) == (ordinals + nominals)
        if ordinals == 0:
            result = self.F[gene][nominalIndex][0]
        else:
            A = self.constructA(self.F[gene][nominalIndex]) # implement the equation 4;do not input A
            result= self.f(A, X_ordinal);  # implement the equation 3
        # print('fitness of gene {} = {}'.format(gene, result))
        return result

    def evaluate(self, individual):
        assert len(individual) == self.N
        totalFitness = 0
        for gene in range(self.N):
            totalFitness += self.fitnessOfGene(individual, gene)
        totalFitness = totalFitness / self.N
        return totalFitness

    ########### BruteForce ##################################################

    def _graydecode(self, gray):
        bin = 0
        while gray:
            bin = bin ^ gray
            gray = gray >> 1
        return bin

    def _grayencode(self, b): # binary to gray
        assert isinstance(b, int)
        return b ^ (b >> 1)

    def _power(self, x, y):
        if x == 0:
            return 0
        if y == 0:
            return 1
        result = x
        for i in range(1, y):
            result = int(result * x)
        return result

    def bruteForceFitness(self, individual, gene):
        nominalIndex = 0
        ordinalIndex = 0
        nominals = 0
        ordinals = 0

        ## get nominalIndex
        # check gene itself's type
        if self.typeOf[gene] == self.variableType['NOMINAL']:
            nominals += 1
            nominalIndex = int(individual[gene])
        else:
            ordinals += 1
            ordinalIndex = int(individual[gene])
        # check neighbour's type
        for k in range(self.K):
            neighbour = self.E[gene][k]
            multiplier = 0
            if self.typeOf[neighbour] == self.variableType['NOMINAL']:
                multiplier = self._power(self.L, nominals)
                nominalIndex += int(multiplier * individual[neighbour])   # check whether need to convert to int ?
                nominals += 1
            else:
                multiplier = self._power(2, ordinals)
                ordinalIndex += int(multiplier * individual[neighbour])
                ordinals += 1

        return self.F[gene][nominalIndex][ordinalIndex]

    def bruteForceEvaluate(self, individual):
        totalFitness = 0
        for gene in range(self.N):
            totalFitness += self.bruteForceFitness(individual, gene)
        totalFitness = totalFitness / self.N
        return totalFitness

    def bruteForce(self, bestFitness, individual, gene):
        if gene == self.N:
            fitness = self.bruteForceEvaluate(individual)
            if self.logVerbose == True:
                self.logger.info('individual: {}'.format(individual))
                self.logger.info('--> {}'.format(fitness))

            realIndex = 0
            intIndex = 0
            nominalIndex = 0
            index = 0
            Nr = 0
            Nz = 0
            Nd = 0
            for i in range(self.N):
                if self.typeOf[i] == self.variableType['REAL']:
                    realIndex += int(individual[i] * math.pow(2, Nr))
                    index += int(individual[i] * math.pow(2,Nr+Nz))
                    Nr += 1 # check this statement and post-increment operator
                elif self.typeOf[i] == self.variableType['INTEGER']:
                    intIndex += int(individual[i] * math.pow(2, Nz))
                    index += int(individual[i] * math.pow(2, Nr+Nz))
                    Nz += 1
                elif self.typeOf[i] == self.variableType['NOMINAL']:
                    nominalIndex += int(individual[i] * math.pow(2, Nd))
                    Nd += 1
            if self.logVerbose == True:
                self.logger.info('* D: nominalIndex = {}, grayencode(realIndex) = {}, grayencode(intIndex) = {}, fitness = {}'.format(nominalIndex, self._grayencode(realIndex), self._grayencode(intIndex), fitness))
                self.logger.info('* Z: grayencode(index) = {}, fitness = {}'.format(self._grayencode(index), fitness))

            if fitness < self.bestFitness:
                self.bestFitness = fitness
            if self.logVerbose == True:
                self.logger.info("the current best fitness = {}".format(self.bestFitness))
        else:
            max_value = 2
            if self.typeOf[gene] == self.variableType['NOMINAL']:
                max_value = self.L
            for i in range(max_value):
                individual[gene] = i
                self.bruteForce(self.bestFitness, individual, gene+1)

    def runBruteForce(self):
        self.individual = [0] * self.N # here initialize self.individual for bruteForce
        self.bruteForce(self.bestFitness, self.individual, 0)
        return self.bestFitness

