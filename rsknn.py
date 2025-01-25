#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Radius Shrinkage For Adaptive Nearest Neighbors Classification

@author: Alexandre L. M. Levada

Python script with the implementation of the proposed RSk-NN

"""

# Imports
import os
import sys
import time
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from scipy import stats
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon

# Computes the curvatures of all samples in the training set
def Curvature_Estimation(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]    
    # First fundamental form
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity', include_self=False)
    A = knnGraph.toarray()    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        #  Computation of the first fundamental form
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)      # isolated points
        # Compute the eigenvectors
        v, w = np.linalg.eig(I)
        # Sort the eigenvalues
        ordem = v.argsort()
        # Select the eigenvectors in decreasing order (in columns)
        Wpca = w[:, ordem[::-1]]
        # Computation of the second fundamental form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Add a column of ones
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))
        Q = Wpca
        # Discard the first m columns of H
        H = Q[:, (m+1):]        
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        curvatures[i] = abs(np.linalg.det(S))
    return curvatures

# Computes the curvature of a single point (test sample)
def Point_Curvature_Estimation(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # First fundamental form 
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvature = 0
    amostras = dados
    ni = n
    # Computation of the first fundamental form
    I = np.cov(amostras.T)
    # Compute the eigenvectors
    v, w = np.linalg.eig(I)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the eigenvectors in decreasing order (in columns)
    Wpca = w[:, ordem[::-1]]
    # Computation of the second fundamental form
    for j in range(0, m):
        Squared[:, j] = Wpca[:, j]**2
    col = 0
    for j in range(0, m):
        for l in range(j, m):
            if j != l:
                Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                col += 1
    # Add a column of ones
    Wpca = np.column_stack((np.ones(m), Wpca))
    Wpca = np.hstack((Wpca, Squared))
    Wpca = np.hstack((Wpca, Cross))
    Q  = Wpca
    # Discard the first m columns of H        
    H = Q[:, (m+1):]
    II = np.dot(H, H.T)
    S = -np.dot(II, I)
    curvature = abs(np.linalg.det(S))
    return curvature

# Optional function to normalize the curvatures to the interval [a, b]
def normalize_curvatures(curv, a, b):
    k = a + (b - a)*(curv - curv.min())/(curv.max() - curv.min())
    return k

# Generates the k-NNG (fixed k)
def Simple_Graph(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # Generate k-NN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance', include_self=False)
    A = knnGraph.toarray()
    return A

# Generates the adaptive k-NNG (different k for each sample)
def Curvature_Based_Graph(dados, k, ncurv):
    n = dados.shape[0]
    m = dados.shape[1]
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=n, mode='distance', include_self=False)
    A = knnGraph.toarray()
    percentiles = np.percentile(A, ncurv, axis=1)
    # Se distância entre xi e xj é maior que o percentil p, desconecta do grafo
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] <= percentiles[i]:
                A[i, j] = 1
            else:
                A[i, j] = 0
    return A

####################################
# Regular k-NN classifier
####################################

# Test regular k-NN classifier
def testa_KNN(treino, teste, target_treino, target_teste, nn):
    n = teste.shape[0]
    m = teste.shape[1]
    labels = np.zeros(len(target_teste))
    for i in range(n):
        data = np.vstack((treino, teste[i, :]))
        rotulos = np.hstack((target_treino, target_teste[i]))
        knnGraph = sknn.kneighbors_graph(data, n_neighbors=nn, mode='distance', include_self=False)
        A = knnGraph.toarray()
        vizinhos = A[-1, :]                 # last line of the adjacency matrix
        indices = vizinhos.nonzero()[0]
        labels[i] = stats.mode(rotulos[indices])[0]
        del data
        del rotulos
    return labels

##############################################
# Radius shrinkage curvature based kk-NN classifier
##############################################

# Test the adaptive radius shrinkage kk-NN classifier
def testa_curvature_KNN(treino, teste, target_treino, target_teste, nn):
    n = teste.shape[0]
    m = teste.shape[1]
    labels = np.zeros(len(target_teste))
    curvaturas = Curvature_Estimation(treino, nn)
    for i in range(n):
        # Computes the nearest neighbors of the i-th test sample
        nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(treino)
        distances, neighs = nbrs.kneighbors(teste[i, :].reshape(1, -1))
        neighs = neighs[0]
        distances = distances[0]
        # Test sample + k nearest neighbors
        data = np.vstack((teste[i, :], treino[neighs, :]))  # add sample at the beginning
        Ki = Point_Curvature_Estimation(data, nn)
        # Add curvature in the vector of curvsatures
        curvaturas_ = np.hstack((curvaturas, Ki))
        # if the curvatures are not all zero
        if curvaturas_.any():
            curvaturas_ = np.log(curvaturas_)
            minimo = np.nanmin(curvaturas_[curvaturas_ != -np.inf])
            curvaturas_ = np.where(curvaturas_ == -np.inf, minimo, curvaturas_)
            ncurv = normalize_curvatures(curvaturas_, 0, 1)
            shrinkage = 1 - ncurv  # se curvatura é alta, fator de shrinkage é pequeno para diminuir bem o raio
        else:
            shrinkage = 1 - curvaturas_
        # Add the test sample
        data = np.vstack((treino, teste[i, :]))
        rotulos = np.hstack((target_treino, target_teste[i]))
        # Build the k-NN graph
        knnGraph = sknn.kneighbors_graph(data, n_neighbors=data.shape[0]-1, mode='distance', include_self=False)
        W = knnGraph.toarray()
        percentiles = np.zeros(len(shrinkage))
        # Compute the radius
        raio = W[-1, W[-1, :].argsort()[nn]]    # raio é a distância até a nn-ésima amostra mais próxima (nn é um parâmetro)
        percentile = np.percentile(W[-1, :], raio*shrinkage[-1])                
        # Se distância entre xi e xj é maior que o percentil p, desconecta do grafo
        A = W.copy()
        for k in range(A.shape[1]):
            if A[-1, k] > 0 and A[-1, k] <= percentile:
                A[-1, k] = 1
            else:
                A[-1, k] = 0
        vizinhos = A[-1, :]                 # last line of the adjacency matrix
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:
            labels[i] = rotulos[np.argmin(W[-1, np.nonzero(W[-1, :])])]  # rótulo do vizinho mais próximo
        else:
            labels[i] = stats.mode(rotulos[indices])[0]
        del data
        del rotulos    
    return labels


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

############################################################
# Data loading (uncomment one dataset from the list below)
############################################################

X = skdata.fetch_openml(name='vowel', version=1)   
#X = skdata.fetch_openml(name='zoo', version=1)    
#X = skdata.fetch_openml(name='thyroid-new', version=1)
#X = skdata.fetch_openml(name='analcatdata_lawsuit', version=1)  
#X = skdata.fetch_openml(name='arsenic-male-bladder', version=2) 
#X = skdata.fetch_openml(name='prnn_crabs', version=1)   
#X = skdata.fetch_openml(name='monks-problems-1', version=1)   
#X = skdata.fetch_openml(name='diggle_table_a2', version=1)  
#X = skdata.fetch_openml(name='user-knowledge', version=1)   
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)                    
#X = skdata.fetch_openml(name='parkinsons', version=1)   
#X = skdata.fetch_openml(name='glass', version=1)            
#X = skdata.fetch_openml(name='biomed', version=1)      
#X = skdata.fetch_openml(name='kidney', version=2)      
#X = skdata.fetch_openml(name='penguins', version=1)      
#X = skdata.fetch_openml(name='sa-heart', version=1) 
#X = skdata.fetch_openml(name='veteran', version=2)  
#X = skdata.fetch_openml(name='arsenic-female-bladder', version=2)
#X = skdata.fetch_openml(name='dbworld-subjects', version=1)
#X = skdata.fetch_openml(name='fl2000', version=2) 
#X = skdata.fetch_openml(name='fri_c2_100_10', version=2) 
#X = skdata.fetch_openml(name='Touch2', version=1) 
#X = skdata.fetch_openml(name='prnn_viruses', version=1) 
#X = skdata.fetch_openml(name='confidence', version=2) 
#X = skdata.fetch_openml(name='Speech', version=1)  

### Large datasets
#X = skdata.fetch_openml(name='wall-robot-navigation', version=1)  
#X = skdata.fetch_openml(name='letter', version=1)              
#X = skdata.fetch_openml(name='pendigits', version=1)
#X = skdata.fetch_openml(name='MNIST_784', version=1)
#X = skdata.fetch_openml(name='Indian_pines', version=1)        
#X = skdata.fetch_openml(name='gas-drift', version=1)           
#X = skdata.fetch_openml(name='JapaneseVowels', version=1)
#X = skdata.fetch_openml(name='USPS', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='11_Tumors', version=1)
#X = skdata.fetch_openml(name='Lung', version=1) 

### High dimensional datasets
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1) 
#X = skdata.fetch_openml(name='micro-mass', version=2)
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)
#X = skdata.fetch_openml(name='OVA_Lung', version=1) 
#X = skdata.fetch_openml(name='OVA_Endometrium', version=1) 

dados = X['data']
target = X['target']

# Reduce large datasets
if dados.shape[0] > 20000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.05, random_state=42)
elif dados.shape[0] > 10000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.1, random_state=42)
elif dados.shape[0] > 2500:
  dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42)

# Convert labels to integers
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy
    dados = dados.to_numpy()
    le = LabelEncoder()
    le.fit(target)
    target = le.transform(target)

# Remove nan's
dados = np.nan_to_num(dados)

if dados.shape[1] > 100:
    model = PCA(n_components=10)
    dados = model.fit_transform(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

n = dados.shape[0]
m = dados.shape[1]
# Number of neighbors
nn = round(np.log2(n))  
# if even, add 1 to become odd
if nn % 2 == 0:
    nn += 1

# Number of classes
c = len(np.unique(target))

print('Number of samples = ', n)
print('Number of features = ', m)
print('Number of classes = %d' %c)
print('Number of neighbors = %d' %nn)
print()

treino_sizes = [0.5]

matriz_knn = np.zeros((len(treino_sizes), 7))   # 7 performance evaluation metrics
matriz_kknn = np.zeros((len(treino_sizes), 7))  # 7 performance evaluation metrics

inicio = time.time()

for i, size in enumerate(treino_sizes):
    print('********************************************')
    print('********** Size of training set: %.2f' %size)
    print('********************************************')
    print()

    treino, teste, target_treino, target_teste = train_test_split(dados, target, train_size=size, random_state=42)

    # regular k-NN 
    rotulos_ = testa_KNN(treino, teste, target_treino, target_teste, nn)
    acc_ = accuracy_score(target_teste, rotulos_)
    bal_acc_ = balanced_accuracy_score(target_teste, rotulos_)
    f1_ = f1_score(target_teste, rotulos_, average='weighted')
    kappa_ = cohen_kappa_score(target_teste, rotulos_)
    prec_ = precision_score(target_teste, rotulos_, average='weighted')
    rec_ = recall_score(target_teste, rotulos_, average='weighted')
    jac_ = jaccard_score(target_teste, rotulos_, average='weighted')

    print('Regular KNN')
    print('-------------')
    print('Balanced accuracy:', bal_acc_)
    print('F1 score:', f1_)
    print('Kappa:', kappa_)
    print('Precision:', prec_)
    print('Recall:', rec_)
    print('Jaccard:', jac_)

    # Adaptive curvature based kk-NN
    rotulos = testa_curvature_KNN(treino, teste, target_treino, target_teste, nn)
    acc = accuracy_score(target_teste, rotulos)
    bal_acc = balanced_accuracy_score(target_teste, rotulos)
    f1 = f1_score(target_teste, rotulos, average='weighted')
    kappa = cohen_kappa_score(target_teste, rotulos)
    prec = precision_score(target_teste, rotulos, average='weighted')
    rec = recall_score(target_teste, rotulos, average='weighted')
    jac = jaccard_score(target_teste, rotulos, average='weighted')

    print()
    print('Radius Shrinkage KNN')
    print('-------------------_-')
    print('Balanced accuracy:', bal_acc)
    print('F1 score:', f1)
    print('Kappa:', kappa)
    print('Precision:', prec)
    print('Recall:', rec)
    print('Jaccard:', jac)
    print()

    measures_knn = np.array([acc_, bal_acc_, f1_, kappa_, prec_, rec_, jac_])
    measures_curvature_knn = np.array([acc, bal_acc, f1, kappa, prec, rec, jac])

    matriz_knn[i, :] = measures_knn
    matriz_kknn[i, :] = measures_curvature_knn

fim = time.time()

print('Elapsed time : %f s' %(fim-inicio))
print()