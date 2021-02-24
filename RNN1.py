#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:07:09 2021

@author: theophilebaggio
"""

#expressivité des reseaux neuronaux
import numpy as np
from math import *
import random as rd
import matplotlib.pyplot as plt

def activate(x,W,b):
    M = 1/(1 + np.exp(-(W.dot(x) + b)))
    return M

def cost_function(W1,W2,b1,b2,X,Y):
    costvec = np.zeros(len(X[0]))
    for i in range(0,len(X[0])):
        x = X[:,i].reshape(len(X),1)
        y = Y[:,i].reshape(len(Y),1)
        a1 = activate(x,W1,b1)
        a2 = activate(a1,W2,b2)
        costvec[i] = np.linalg.norm(y-a2,2)
    costvalue = 1/len(costvec) * 1/2 * np.linalg.norm(costvec,2)**2
    return costvalue
        
##data
donnees = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],\
                    [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]])
target = np.array([[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1]])

##initialisation des paramètres du NN
nbhidden1 = 4 #nb de neurones dans la hidden layer
nbinput = 2
nboutput = 2
#ajouter un seed
Weight_mat1  = 1/2 * np.random.randint(10, size=(nbhidden1,nbinput),)
bias_1 =1/2 * np.random.randint(10, size=(nbhidden1,1))
Weight_mat2 =1/2 * np.random.randint(10, size=(nboutput,nbhidden1))
bias_2= 1/2 * np.random.randint(10, size=(nboutput,1))

##initialisation des paramètres du gradient stochastique
eta = 0.05 #learnig rate
Niter = int(1e6) #nombre d'iterations maximum
savecost = np.zeros((Niter))
for i in range(0,Niter):
    
    #forward propagation
    k = np.random.randint(len(donnees[0]))
    x = donnees[:,k].reshape(nbinput,1); #choix de input
    a1 = activate(x,Weight_mat1,bias_1)
    da1= a1*(1-a1)
    matda1 = np.diag(da1[:,0])
    a2 = activate(a1,Weight_mat2,bias_2)
    da2 = a2*(1-a2)
    matda2 = np.diag(da2[:,0])
    
    #backpropagation
    delta2 = matda2.dot(a2-target[:,k].reshape(nboutput,1))
    delta1 = matda1.dot(np.transpose(Weight_mat2)).dot(delta2)
    
    #on ajuste les coefficients
    Weight_mat2 = Weight_mat2 - eta*delta2.dot(np.transpose(a1))
    bias_2 = bias_2 - eta*delta2
    Weight_mat1 = Weight_mat1 - eta*delta1.dot(np.transpose(x))
    bias_1= bias_1 - eta*delta1
    
    #on sauve la valeur de la cost function après chaque itération de
    #la méthode du gradient stochastique
    savecost[i] = cost_function(Weight_mat1,\
                                Weight_mat2, bias_1, bias_2, donnees, target)

plt.semilogy(savecost)
plt.xlabel("nombre d'itérations")
plt.ylabel('Cost function')
plt.title("Optimisation d'un RN avec %i neurones dans l'unique couche cachée"\
          %nbhidden1)


