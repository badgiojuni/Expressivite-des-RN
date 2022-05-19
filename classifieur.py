#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:53:40 2021

@author: theophilebaggio
"""


import numpy as np
from math import *
import random as rd
import matplotlib.pyplot as plt

def activate(x,W,b):
    M = 1/(1 + np.exp(-(W.dot(x) + b)))
    return M

def cost_function(Weight_mat_list,bias_list,X,Y):
    costvec = np.zeros(len(X[0]))
    for i in range(0,len(X[0])):
        x = X[:,i].reshape(len(X),1)
        y = Y[:,i].reshape(len(Y),1)
        a = x
        for j in range(len(bias_list)):
            a = activate(a,Weight_mat_list[j],bias_list[j])
        costvec[i] = np.linalg.norm(y-a,2)
    costvalue = 1/len(costvec) * 1/2 * np.linalg.norm(costvec,2)**2
    return costvalue

def predict(Weight_mat_list,bias_list,X):
    prediction_list = []
    for i in range(len(X[0])):
        a = X[:,i].reshape(len(X),1)
        for j in range(len(bias_list)):
            a = activate(a,Weight_mat_list[j],bias_list[j])
        if(a[0]<=a[1]):
            prediction_list.append([0,1])
        else: 
            prediction_list.append([1,0])
    return np.array(prediction_list).T
            
    
## MAIN PROGRAM
#data
donnees = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7,0.6,0.1],\
                    [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6,0.5,1.5]])
target = np.array([[1,1,1,1,1,0,0,0,0,0,1,0],[0,0,0,0,0,1,1,1,1,1,0,1]])

nb_layers = 1
liste_nb_neurons = []
liste_weigth_mat = []
liste_bias = []

for i in range(nb_layers):
    n = i+1
    neurons = int(input('How many neurons would you like\
 in the hidden layer n° {} ?'.format(n)))
    liste_nb_neurons.append(neurons)

nb_n_hid_lay = sum(liste_nb_neurons)
nbinput = 2
nboutput = 2  
liste_nb_neurons.append(nboutput)
liste_nb_neurons.insert(0,nbinput)
 
for nb in range(nb_layers+1):
    n_current_layer = liste_nb_neurons[nb+1]
    n_previous_layer = liste_nb_neurons[nb]#nb de neurones dans la hidden layer
    np.random.seed(5);
    weight_mat  = 1/2 * np.random.randint(10,\
                                           size=(n_current_layer,n_previous_layer))
    bias=1/2 * np.random.randint(10, size=(n_current_layer,1))
    liste_weigth_mat.append(weight_mat)
    liste_bias.append(bias)
    
##initialisation des paramètres du gradient stochastique
eta = 0.05 #learnig rate
Niter = int(1e6) #nombre d'iterations maximum
savecost = np.zeros((Niter))

for i in range(0,Niter):
    
    #feedforward
    k = np.random.randint(len(donnees[0]))
    x = donnees[:,k].reshape(nbinput,1) 
    activation_vect = [x]
    d_activation_vect = []
    mat_da_vect = []
    
    for j in range(nb_layers + 1):
        a = activate(activation_vect[j],liste_weigth_mat[j],\
                                        liste_bias[j])
        activation_vect.append(a)
        d_activation_vect.append(a*(1-a))
        mat_da_vect.append(np.diag(d_activation_vect[j][:,0]))
    
    #backpropagation
    liste_delta = []
    delta_out = mat_da_vect[-1].dot\
        ((activation_vect[-1]-target[:,k].reshape(nboutput,1)))
    liste_delta.append(delta_out)
    
    for j in range(nb_layers):
        delta = mat_da_vect[-1 -(j+1)].dot\
            (np.transpose(liste_weigth_mat[-1 - j]).dot(liste_delta[j]))
        liste_delta.append(delta)
    liste_delta.reverse()
    
    #on ajuste les coefficients
    for j in range(len(liste_delta)):
        a = activation_vect[-1-(j+1)]
        prov_value = liste_delta[-1-j].dot(np.transpose(a))
        mat = liste_weigth_mat[-1-j]
        liste_weigth_mat[-1-j] = mat - eta*prov_value
        liste_bias[-1-j] = liste_bias[-1-j]- eta*liste_delta[-1-j]
    
    #on sauve la valeur de la cost function après chaque itération de
    #la méthode du gradient stochastique
    savecost[i] = cost_function(liste_weigth_mat,\
                               liste_bias,  donnees, target)

predictions = predict(liste_weigth_mat,liste_bias,donnees)
print(predictions)
    
plt.figure(2)
plt.semilogy(savecost)
plt.xlabel("nombre d'itérations")
plt.ylabel('Cost function')
plt.title("Optimisation d'un RN avec {} neurones dans\
 la couche cachée".format(nb_n_hid_lay))
axes = plt.gca()
 
#on essaie de dessiner la séparation par zone apprise par le reseau neuronal
x1 = donnees[0]
x2 = donnees[1]
y = target
h = 0.01
x_min, x_max = x1.min() - 1, x1.max() + 1
y_min, y_max = x2.min() - 1, x2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = predict(liste_weigth_mat,liste_bias,np.c_[xx.ravel(), yy.ravel()].T)

plt.figure(3)
for i in range(len(x1)):
        if(y[1,i]==1):
            plt.plot(x1[i],x2[i],'bx')
        else:
            plt.plot(x1[i],x2[i],'ro')

Z = Z[0].reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.title("Région de décision obtenue pour {} itérations".format(Niter))
plt.show()


