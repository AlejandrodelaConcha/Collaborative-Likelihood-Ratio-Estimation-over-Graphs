# -----------------------------------------------------------------------------------------------------------------
# Title:  likelihood_ratio_collaborative
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-25              
# Current version:  2025-02-25
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): 
# This code implements GRULSIF and POOL as described in the paper 
# "Collaborative Likelihood Ratio Estimation."
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, numba, joblib, Models.aux_functions
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GRULSIF_implementation, Pool_implementation
# -----------------------------------------------------------------------------------------------------------------

import numpy as np
import copy
import sys
from numpy import linalg as LA
from Models.aux_functions import *
from numba import njit,jit
from numba.typed import List
from scipy.sparse.linalg import eigsh
from scipy import sparse
from joblib import Parallel, delayed
from itertools import product


class SymmetryError(Exception):
    """A custom exception class with a message."""
    def __init__(self, message="The matrix is not symetric"):
        self.message = message
        super().__init__(self.message)


def cost_function(theta,H,h_test): 
####### The cost function is the sum of the least square error for each of the nodes.

## Input
# theta: numpy matrix of dimension n_nodesxL each row is the estimated parameter for the node v 
# H: numpy matrix of dimension n_nodesxLxL, the first coordinates indicates a matrix of (1-alpha)*K()K()^T+alpha*K()K()^T over observations from the datapoints comming from p_v and q_v
# h_test: numpy matrix of dimension n_nodesxL each row is a vector with means of K() over observations from the distribution q_v(.)
# alpha: regularization parameter associated with the upperbound of the relative likelihood ratio it should be between (0,1)

## Output
# Returns the cost-function  $ \frac{1}{n_nodes} \sum_{v \in V } (\frac{1}{2} \theta_v^{\top}( \alpha H'_v +(1-\alpha) H_v   \theta_v - \theta_v^{\top} h'_v)

    n_nodes=len(H)
    cost=np.zeros(n_nodes)
    thetas=copy.deepcopy(theta)
    for i in range(n_nodes):
        cost[i]=thetas[i].dot(H[i]).dot(thetas[i])
        cost[i]*=0.5
        cost[i]-=thetas[i].dot(h_test[i])
    return np.mean(cost)

def update_hs(ref_index,test_index,data_ref,data_test,kernel,alpha):  
########### This function estimates the variables H h_test for each of the parameters for the estimation of GRULSIF
    
## Input
# ref_index= indexes to be used in the estimation of the feature vectors for the points in the reference set 
# test_index: indexes to be used in the estimation of the feature vectors for the points in the test set 
# both data_ref and data_test are lists with n_nodes elements and each of them a numpy array of dimension (n_v,L), where n_v is the number of observations at node v
# data_ref: data points representing the distribution p_v(.)
# data_test: data points representing the distribution q_v(.)
# kernel: the kernel function being used for estimation 
# alpha: regularization parameter associated with the upperbound of the relative likelihood ratio it should be between (0,1)

## Output
# H: numpy matrix of dimension n_nodesxLxL, the first coordinates indicates a matrix of (1-alpha)*K()K()^T+alpha*K()K()^T over observations from the datapoints comming from p_v and q_v
# h_test: numpy matrix of dimension n_nodesxL each row is a vector with means of K() over observations from the distribution q_v(.)

    n_nodes=len(data_ref) 
    L=kernel.n

    ############# Estimate H

    h_test =np.zeros((n_nodes,L),dtype=np.float32)
    H = np.zeros((n_nodes,L,L),dtype=np.float32)
    for i in range(n_nodes):
        phi_test=kernel.k_V(data_test[i][test_index])
        phi_ref=kernel.k_V(data_ref[i][ref_index])
        N_ref=len(phi_ref)
        N_test=len(phi_test)
        h_test[i] = np.sum(phi_test, axis=0)
        h_test[i]/= N_test
        H[i]=np.einsum('ji,j...',  phi_ref,phi_ref)
        H[i]*=(1-alpha)/N_ref
        H[i]+=np.einsum('ji,j...',  phi_test,phi_test)*(alpha/N_test)
        
    return H,h_test


def find_dictionary(threshold_coherence,sigma,data_ref,data_test,kernel_type="Gaussian",verbose=False):
##### This function implements the method described in Richard et al. (2009) across all the nodes 

## Input
# threshold_coherence: parameter related to the dictionary selection described in Richard et al. (2009) 
#                      When the kernel is normal, this parameter should be between 0 and 1.
#                      The closer it is to 1, the larger the dictionary and the slower the training.
# sigma: hyperparameter related to the kernel. As we only work with the Laplace and Gaussian kernel, 
#        sigma is the width parameter.
# kernel_type: ["Gaussian", "Laplace"]
# verbose: whether or not to print the count of the elements in the dictionary. 

## Output
# dictionary: the selected dictionary via the threshold coherence.

 
     n_nodes=len(data_ref) 
     d=data_ref[0].shape[1]
     
     if verbose:
         print("Initializing the dictionary")
     
     dictionary=[]
     first_element=np.array(data_test[0][0]).reshape(1,d)
     if kernel_type=="Gaussian":
         kernel_1=Gauss_Kernel(first_element,sigma)
     elif kernel_type=="Laplace":
         kernel_1=Laplace_Kernel(first_element,sigma)
     
  
     for i in range(n_nodes):
         for t in range(len(data_test[i])):
             _,coherence = kernel_1.coherence(np.atleast_2d(data_test[i][t]))
             if coherence < threshold_coherence:
                 kernel_1.add_dictionary(np.atleast_2d(data_test[i][t]))
                 
         for t in range(len(data_ref[i])):
             _,coherence = kernel_1.coherence(np.atleast_2d(data_ref[i][t]))
             if coherence < threshold_coherence:
                 kernel_1.add_dictionary(np.atleast_2d(data_ref[i][t])) 
                                 
     dictionary.extend(kernel_1.dictionary)
     dictionary=np.vstack(dictionary)
     np.random.shuffle(dictionary)
            
     if verbose:
         print("Total size")
         print(kernel_1.n)      
                
     return dictionary



def score(theta,H,h_test):
### Function estimating the Pearson Divergence at the node level 

## Input
# theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
# H: numpy matrix of dimension n_nodes x L x L; it represents n_nodes matrices 
#    (1 - alpha) * K() K()^T + alpha * K() K()^T over observations from the data points coming from p_v and q_v.
# h_test: numpy matrix of dimension n_nodes x L; each row is a vector with means of K() 
#         over observations from the distribution q_v(.).
# alpha: regularization parameter associated with the upper bound of the relative likelihood ratio; 
#        it should be between (0,1).

## Output 
# score: a numpy vector of dimension n_nodes; each element represents the Pearson Divergence at the node level.
    
    n_nodes=len(H)
    score=np.zeros(n_nodes)
    thetas=copy.deepcopy(theta)
    for i in range(n_nodes):
        score[i]=(thetas[i].dot(H[i])).dot(thetas[i])
        score[i]*=-0.5
        score[i]+=thetas[i].dot(h_test[i])
        score[i]-=0.5 
    return score

@jit(nopython=True)
def optimize_GRULSIF(theta_ini,W,cols,row_pointer,A,h_test,learning_rates,alpha,gamma,lamb,tol=1e-2,verbose=False):
### Function running the optimization scheme for updating the parameters when the graph is used

## Input
# theta_ini: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
# W: the weights of the adjacency matrix, stored optimally to save memory.
# cols, row_pointer: refer to the localization of the nonzero entries.
# A: numpy matrix of dimension n_nodes x L x L; each entry is defined as 
#    (alpha * H'_v + (1 - alpha) * H_v) / n_nodes + lamb * self.degrees[i].
# h_test: numpy matrix of dimension n_nodes x L; each row is a vector with means of K() 
#         over observations from the distribution q_v(.).
# learning_rates: a vector of dimension n_nodes, where each entry corresponds to a learning rate of the CBGD.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# gamma: regularization parameter related to the norm of the function f_v.
# lamb: regularization parameter related to graph smoothness.
# neighbours: neighbors associated with each of the nodes.
# tol: level of accepted tolerance in the estimation; should be between 0 and 1, closer to zero.
# verbose: whether or not to print the estimation error at each iteration.

## Output
# theta_new: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.

    n_nodes=len(theta_ini)
    L = len(theta_ini[0]) 
    theta_old=1*theta_ini
    theta_new=1*theta_ini
    k_=0
    error=np.inf
    theta_old_vector = 1.*theta_new
    while error>tol:
        k_+=1
       # theta_old_vector=1*theta_old
        for i in range(n_nodes):
            aux_b = np.zeros(L,dtype=np.float32) 
            start_idx = row_pointer[i]
            end_idx = row_pointer[i+ 1]
            for j in range(start_idx,end_idx):
                aux_b -= W[j]*theta_old[cols[j]]
            aux_b *= lamb
            aux_b -= h_test[i]/n_nodes
            gradient = A[i].dot(theta_old[i])
            gradient+= aux_b
            theta_new[i] =learning_rates[i]*theta_old[i]-gradient
            theta_new[i]/=(gamma+learning_rates[i])
            theta_old[i]=1*theta_new[i]
        error=np.linalg.norm(theta_old_vector-theta_new)/np.linalg.norm(theta_old_vector)
        if verbose:
            print(k_)
            print(error)
        theta_old_vector = 1.*theta_new
     
    return theta_new



@jit(nopython=True)
def optimize_Pool(theta_ini,A,h_test,learning_rates,alpha,gamma,tol=1e-2,verbose=False):
####### Function running the optimization scheme for updating the parameters when the graph is ignored

## Input
# theta_ini: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
# A: numpy matrix of dimension n_nodes x L x L; each entry is defined as 
#    (alpha * H_test + (1 - alpha) * H_ref) / n_nodes.
# h_test: numpy matrix of dimension n_nodes x L; each row is a vector with means of K() 
#         over observations from the distribution q_v(.).
# learning_rates: a vector of dimension n_nodes, where each entry corresponds to a learning rate of the CBGD.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# gamma: regularization parameter related to the norm of the function f_v.

# tol: acceptable tolerance level in the estimation; should be between 0 and 1, closer to zero.
# verbose: whether or not to print the estimation error at each iteration.

## Output
# theta_new: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.

    L = len(theta_ini[0])
    n_nodes=theta_ini.shape[0]
    theta_old=1*theta_ini
    theta_new=1*theta_ini
    k_=0
    error=np.inf
    theta_old_vector = 1.*theta_new
    while error>tol:
        k_+=1
        theta_old_vector=1*theta_old
        for i in range(n_nodes):
            aux_b = - h_test[i]/n_nodes
            gradient = A[i].dot(theta_old[i])
            gradient+= aux_b
            theta_new[i]=learning_rates[i]*theta_old[i]-gradient
            theta_new[i]/=gamma+learning_rates[i]
            theta_old[i]=1*theta_new[i]
            
        error=np.linalg.norm(theta_old_vector-theta_new)/np.linalg.norm(theta_old_vector)
        if verbose:
            print(k_)
            print(error)
        theta_old_vector = 1.*theta_new
     
    return theta_new


def CROSS_validation_GRULSIF(k_cross_validation,data_ref,data_test,W,cols,row_pointer,degrees,alpha,threshold_coherence=0.1,tol=1e-2,sigma_list=None,verbose=False,kernel_type="Gaussian"):
    
##### This function initializes the hyperparameters of the problem via cross-validation , that is the 
### sigma related with each of the kernels, the asociated dictionary; the parameter gamma of sparseness at the node level 
### the lambda parameter related with the Laplacian regularization

## Input
# k_cross_validation: number of splits for cross-validation.
# data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
#                         where L is the dimension of the feature space.
# data_ref[v]: data points representing the distribution p_v(.).
# data_test[v]: data points representing the distribution q_v(.).
# W: the weights of the adjacency matrix, stored optimally to save memory.
# cols, row_pointer: refer to the localization of the nonzero entries.
# degrees: vector containing the degrees of each of the nodes.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
#                      When the kernel is normal, this parameter should be between 0 and 1.
#                      The closer it is to 1, the larger the dictionary and the slower the training.
# tol: acceptable tolerance level in the estimation.
# sigma_list: list of candidate sigma values.
# verbose: whether or not to print intermediate steps.
# kernel_type: the type of kernel used in the optimization method: "Gaussian" or "Laplacian".

## Output
# kernel_new: A Nystrom kernel class initialized with the optimized sigma parameter and a dictionary.
# gamma_new: optimal regularization parameter related to the norm of the function f_v.
# lamb_new: optimal regularization parameter related to graph smoothness.

    N_ref=np.min([len(data) for data in data_ref])
    N_test=np.min([len(data) for data in data_test])

    n_nodes=len(data_ref)
    
    aux_index_ref=np.arange(N_ref)
    np.random.shuffle(aux_index_ref)
    aux_index_test=np.arange(N_test)
    np.random.shuffle(aux_index_test)
    ref_index_validation=[aux_index_ref[int(i*(N_ref/k_cross_validation)):int((i+1)*((N_ref/k_cross_validation)))] for i in range(k_cross_validation)]
    test_index_validation=[aux_index_test[int(i*(N_test/k_cross_validation)):int((i+1)*((N_test/k_cross_validation)))] for i in range(k_cross_validation)]
  
    ref_index_train=[]
    test_index_train=[]
    
    for i in range(k_cross_validation):
        if i==0:
            ref_index_train.append(np.hstack(ref_index_validation[i+1:]))
            test_index_train.append(np.hstack(test_index_validation[i+1:]))
        elif i==(k_cross_validation-1):
            ref_index_train.append(np.hstack(ref_index_validation[:i]))
            test_index_train.append(np.hstack(test_index_validation[:i]))
        else:
            ref_index_train.append(np.hstack((np.hstack(ref_index_validation[:i]),np.hstack(ref_index_validation[i+1:]))))
            test_index_train.append(np.hstack((np.hstack(test_index_validation[:i]),np.hstack(test_index_validation[i+1:]))))


    gamma_list = np.logspace(-5,1,4)/np.sqrt(n_nodes*np.min((N_ref,N_test)))
    lambda_list = np.logspace(-5, 2, 5)/np.sqrt(n_nodes*np.min((N_ref,N_test)))
    
    cost_matrix=np.zeros((k_cross_validation,len(sigma_list),len( lambda_list),len(gamma_list)))
    learning_rates=np.zeros(n_nodes)
    
    for s in range(len(sigma_list)):
        dictionary=find_dictionary(threshold_coherence,kernel_type=kernel_type,sigma=sigma_list[s],data_ref=data_ref,data_test=data_test,verbose=False)
        L=len(dictionary)
        kernel_=Nystrom_Kernel(dictionary=dictionary,gamma=sigma_list[s],kernel_type=kernel_type)
        theta_ini = 1e-6*np.ones((n_nodes,L),dtype=np.float32)
        
        for k in range(k_cross_validation):
            
            A,h_test_training=update_hs(ref_index_train[k],test_index_train[k],data_ref,data_test,kernel_,alpha=alpha)
          
            for l in range(len(lambda_list)): 
                for i in range(n_nodes):
                    A[i]/=n_nodes
                    A[i]+=lambda_list[l]*degrees[i]*np.eye(L)
                    eta_i, _ = eigsh(A[i],k=1,ncv=np.min((500,L))) 
                    learning_rates[i]=1.*eta_i  
                    
                thetas=[[] for i in range(len(gamma_list))]
                for g in range(len(gamma_list)):
                    thetas[g]=optimize_GRULSIF(theta_ini,W=W,cols=cols,row_pointer=row_pointer,A=A,h_test=h_test_training,learning_rates=learning_rates,
                                                   alpha=alpha,gamma=gamma_list[g],lamb=lambda_list[l],tol=tol,verbose=False)                        
            
                for i in range(n_nodes):
                    A[i]-=lambda_list[l]*degrees[i]*np.eye(L)
                    A[i]*=n_nodes
                   

                ######### In order to spped up the optimization
                theta_ini=thetas[0]
         
                for i in range(n_nodes):              
                    phi_test=kernel_.k_V(data_test[i][test_index_validation[k]])
                    phi_ref=kernel_.k_V(data_ref[i][ref_index_validation[k]])
                    N_ref=len(phi_ref)
                    N_test=len(phi_test)
                    h_test = np.sum(phi_test, axis=0)
                    h_test/= N_test
                    H=np.einsum('ji,j...',  phi_ref,phi_ref)
                    H*=(1-alpha)/N_ref
                    H+=np.einsum('ji,j...',  phi_test,phi_test)*(alpha/N_test)
                    for g in range(len(gamma_list)):
                        cost_matrix[k,s,l,g]+=(0.5*thetas[g][i].dot(H.dot(thetas[g][i]))-thetas[g][i].dot(h_test))/n_nodes

        if verbose:
            print(f"dictionary size::{L}")
            for l in range(len(lambda_list)):
                for g in range(len(gamma_list)):                
                   
                    print(f"sigma::{sigma_list[s]},lambda:{lambda_list[l]},gamma:{gamma_list[g]},score:{np.mean(cost_matrix[:,s,l,g]):.4f}")

    cost_matrix=np.mean(cost_matrix,axis=0)
    index=np.unravel_index(np.argmax(-1*cost_matrix), cost_matrix.shape)
    sigma_new=sigma_list[index[0]]
    lamb_new=lambda_list[index[1]]
    gamma_new=gamma_list[index[2]]
    dictionary=find_dictionary(threshold_coherence,kernel_type=kernel_type,sigma=sigma_new,data_ref=data_ref,data_test=data_test,verbose=False)
    kernel_new=Nystrom_Kernel(dictionary=dictionary,gamma=sigma_new,kernel_type=kernel_type)

    return  kernel_new,lamb_new,gamma_new


def CROSS_validation_Pool(k_cross_validation,data_ref,data_test,alpha,threshold_coherence=0.1,tol=1e-2,sigma_list=None,verbose=False,kernel_type="Gaussian"):
 
##### This function initializes the hyperparameters of the problem via cross-validation , that is the 
### sigma related with each of the kernels, the asociated dictionary; the parameter gamma of sparseness at the node level 


## Input
# k_cross_validation: number of splits for cross-validation.
# data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
#                         where L is the dimension of the feature space.
# data_ref[v]: data points representing the distribution p_v(.).
# data_test[v]: data points representing the distribution q_v(.).
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
#                      When the kernel is normal, this parameter should be between 0 and 1.
#                      The closer it is to 1, the larger the dictionary and the slower the training.
# tol: acceptable tolerance level in the estimation.
# sigma_list: list of candidate sigma values.
# verbose: whether or not to print intermediate steps.
# kernel_type: the type of kernel used in the optimization method: "Gaussian" or "Laplacian".

## Output
# kernel_new: A Nystrom kernel class initialized with the optimized sigma parameter and a dictionary.
# gamma_new: optimal regularization parameter related to the norm of the function f_v.

 
   
    N_ref=np.min([len(data) for data in data_ref])
    N_test=np.min([len(data) for data in data_test])
    n_nodes=len(data_ref)
    
    ref_index_validation=[np.arange(N_ref)[int(i*(N_ref/k_cross_validation)):int((i+1)*((N_ref/k_cross_validation)))] for i in range(k_cross_validation)]
    test_index_validation=[np.arange(N_test)[int(i*(N_test/k_cross_validation)):int((i+1)*((N_test/k_cross_validation)))] for i in range(k_cross_validation)]
    ref_index_train=[]
    test_index_train=[]
    
    for i in range(k_cross_validation):
        if i==0:
            ref_index_train.append(np.hstack(ref_index_validation[i+1:]))
            test_index_train.append(np.hstack(test_index_validation[i+1:]))
        elif i==(k_cross_validation-1):
            ref_index_train.append(np.hstack(ref_index_validation[:i]))
            test_index_train.append(np.hstack(test_index_validation[:i]))
        else:
            ref_index_train.append(np.hstack((np.hstack(ref_index_validation[:i]),np.hstack(ref_index_validation[i+1:]))))
            test_index_train.append(np.hstack((np.hstack(test_index_validation[:i]),np.hstack(test_index_validation[i+1:]))))


 ############## H for the first score


    gamma_list = np.logspace(-5,1,4)/np.sqrt(np.min((N_ref,N_test)))
  
    cost_matrix=np.zeros((k_cross_validation,len(sigma_list),len(gamma_list)))
    learning_rates=np.zeros(n_nodes)

    for s in range(len(sigma_list)):
        dictionary=find_dictionary(threshold_coherence,kernel_type=kernel_type,sigma=sigma_list[s],data_ref=data_ref,data_test=data_test,verbose=False)
        L=len(dictionary)
        kernel_=Nystrom_Kernel(dictionary=dictionary,gamma=sigma_list[s],kernel_type=kernel_type)
        theta_ini = 1e-6*np.ones((n_nodes,L),dtype=np.float32)
       
        for k in range(k_cross_validation):
            
            A,h_test_training=update_hs(ref_index_train[k],test_index_train[k],data_ref,data_test,kernel_,alpha)
        
            for i in range(n_nodes):
                A[i]/=n_nodes
                A[i]+=1e-6*np.eye(L)
                eta_i, _ = eigsh(A[i],k=1,ncv=np.min((500,L)))           
                learning_rates[i]=1*eta_i
                     
            thetas=Parallel(n_jobs=-1,prefer="threads")(delayed(optimize_Pool)(theta_ini,A=A,learning_rates=learning_rates,h_test=h_test_training,alpha=alpha,gamma=g,tol=tol,verbose=False) for g in gamma_list)
            
            for i in range(n_nodes):
                A[i]-=1e-6*np.eye(L)
                A[i]*=n_nodes
                
            for i in range(n_nodes):              
                phi_test=kernel_.k_V(data_test[i][test_index_validation[k]])
                phi_ref=kernel_.k_V(data_ref[i][ref_index_validation[k]])
                N_ref=len(phi_ref)
                N_test=len(phi_test)
                h_test = np.sum(phi_test, axis=0)
                h_test/= N_test
                H=np.einsum('ji,j...',  phi_ref,phi_ref)
                H*=(1-alpha)/N_ref
                H+=np.einsum('ji,j...',  phi_test,phi_test)*(alpha/N_test)
                for g in range(len(gamma_list)):
                    cost_matrix[k,s,g]+=(0.5*thetas[g][i].dot(H.dot(thetas[g][i]))-thetas[g][i].dot(h_test))/n_nodes
                    
            theta_ini=thetas[0]
        if verbose:
            print(f"dictionary size::{L}")
            for g in range(len(gamma_list)): 
                print(f"sigma::{sigma_list[s]},gamma:{gamma_list[g]},score:{np.mean(cost_matrix[:,s,g]):.4f}")

    cost_matrix=np.mean(cost_matrix,axis=0)
    index=np.unravel_index(np.argmax(-1*cost_matrix), cost_matrix.shape)
    sigma_new=sigma_list[index[0]]
    gamma_new=gamma_list[index[1]]
    dictionary=find_dictionary(threshold_coherence,kernel_type=kernel_type,sigma=sigma_new,data_ref=data_ref,data_test=data_test,verbose=False)
    kernel_new=Nystrom_Kernel(dictionary=dictionary,gamma=sigma_new,kernel_type=kernel_type)

    return kernel_new,gamma_new

    


class GRULSIF():
### Class implementing the GRULSIF f-divergence estimation 
    def __init__(self,W,data_ref,data_test,threshold_coherence=0.3,alpha=0.1,tol=1e-2,k_cross_validation=5,verbose=False,kernel_type="Gaussian"):   
    ## Input
    # W: adjacency matrix of the graph of interest in sparse format (List of Lists format).
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
    #                      When the kernel is normal, this parameter should be between 0 and 1.
    #                      The closer it is to 1, the larger the dictionary and the slower the training.
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
    # tol: acceptable tolerance level in the estimation.
    # k_cross_validation: number of splits for cross-validation.   
    # verbose: whether or not to print intermediate steps.
    # kernel_type: the type of kernel used in the optimization method: "Gaussian" or "Laplace".
    
#        lens_data_ref=[len(data) for data in data_ref]
#        lens_data_test=[len(data) for data in data_test]
#        min_len_data_ref=np.min(lens_data_ref)
#        min_len_data_test=np.min(lens_data_test)
#        min_len=np.min((min_len_data_ref,min_len_data_test))

        self.n_nodes=W.shape[0]

        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==self.n_nodes and n_nodes_1==self.n_nodes):
                raise ValueError(F"The datasets should be list with as many elements as the numbers of nodes")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
            
        
        try:
            alpha=float(alpha)
            if not (0.0<=alpha<1):
                raise ValueError(F"Parameter alpha must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
        except TypeError as e:
            print("Error: alpha parameter should be a float")
            sys.exit(1)
            
        try:
            k_cross_validation=int(k_cross_validation)
            if not (1<k_cross_validation):
                raise ValueError(F"The number of samples should be bigger than 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
        try:
            threshold_coherence=float(threshold_coherence)
            if not (0.0<threshold_coherence<1):
                raise ValueError(F"The threshold coherence must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: The threshold coherence parameter should be a float")
            sys.exit(1)
            
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)
            
        try: 
            if kernel_type not in ["Gaussian","Laplace"]: 
                raise ValueError(F"Kernel sould be or Gausian or Laplace")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
        try: 
            Ws=(W+W.T)/2
            if not (abs(Ws-Ws.T)>1e-10).nnz == 0: 
                raise SymmetryError(F"The weight matrix should be symmetric")
            if not (W<0).nnz==0:
                raise ValueError(F"All the elements of the weight matrix should bigger or equal to zero ")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except SymmetryError as e:
            print(f"Error: {e}")
            sys.exit(1)
                

        
        self.verbose=verbose
        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.alpha=alpha
        self.W=W.tocoo()
        self.W_ij =self.W.data
        row_indices = self.W.row
        self.col_indices = self.W.col

        num_rows=self.n_nodes
        self.row_ptr = np.empty(num_rows + 1, dtype=int)
        current_row = 0
        for i in range(len(row_indices)):
            while current_row < row_indices[i]:
                self.row_ptr[current_row + 1] = i
                current_row += 1
        self.row_ptr[-1] = len(row_indices)

        self.degrees = np.sum(self.W,axis=0)
        self.degrees=np.squeeze(np.asarray(self.degrees))
        self.N=sum([len(d) for d in data_ref])+sum([len(d) for d in data_test])
        self.sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])
        self.kernel,self.lamb,self.gamma=CROSS_validation_GRULSIF(k_cross_validation,self.data_ref,self.data_test,
                                                                  W=self.W_ij,cols=self.col_indices,row_pointer=self.row_ptr,
                                                                 degrees=self.degrees,alpha=self.alpha,tol=tol,threshold_coherence=threshold_coherence,
                                                                 sigma_list=self.sigma_list,verbose=self.verbose,kernel_type=kernel_type)
        self.L=len(self.kernel.dictionary)
   
    def fit(self,data_ref=None,data_test=None,tol=1e-3,verbose=False): 
    ### Function estimating the theta parameter associated with the likelihood ratios 
    ### for a given set of observations coming from p_v(.) and q_v(.) 

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # tol: acceptable tolerance level in the estimation.
    # verbose: whether or not to print the iterations.

    ## Output
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)  
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
     
        h_s=self.update_hs(data_ref,data_test)
            

        theta_ini=1e-6*np.ones((self.n_nodes,self.kernel.n),dtype=np.float32)     
        learning_rates=np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            h_s[0][i]/=self.n_nodes
            h_s[0][i]+=self.lamb*self.degrees[i]*np.eye(self.kernel.n)
            eta_i, _ = eigsh(h_s[0][i],k=1,ncv=np.min((500,self.kernel.n)))
            learning_rates[i]=1*eta_i
            
        theta=optimize_GRULSIF(theta_ini,W=self.W_ij,cols=self.col_indices,row_pointer=self.row_ptr,
                               A=h_s[0],h_test=h_s[1],learning_rates=learning_rates,alpha=self.alpha,gamma=self.gamma,
                               lamb=self.lamb,tol=tol,verbose=verbose) 
       
        return theta
    
    def update_hs(self,data_ref=None,data_test=None):
    ####This function estimates the variables H and h_test for each parameter in the estimation of GRULSIF

    ## Input

    # data_ref: data points representing the distribution p_v(.).
    # data_test: data points representing the distribution q_v(.).

    ## Output
    # H: numpy matrix of dimension n_nodes x L x L; the first coordinate indicates a matrix of 
    #    (1 - alpha) * K() K()^T + alpha * K() K()^T over observations from the data points coming from p_v and q_v.
    # h_test: numpy matrix of dimension n_nodes x L; each row is a vector with means of K() 
    #         over observations from the distribution q_v(.).

        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
                
        phi_test=List()
        phi_ref=List()  
        
          
        for i in range(self.n_nodes):
            phi_test.append(self.kernel.k_V(data_test[i]))
            phi_ref.append(self.kernel.k_V(data_ref[i]))     
  
    ############# Estimate H
        h_test =np.zeros((self.n_nodes,self.kernel.n),dtype=np.float32)
        H = np.zeros((self.n_nodes,self.kernel.n,self.kernel.n),dtype=np.float32) 
        for i in range(self.n_nodes):
            N_ref=len(phi_ref[i])
            N_test=len(phi_test[i])
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H[i]*=(1-self.alpha)/N_ref
            H[i]+=np.einsum('ji,j...',  phi_test[i],phi_test[i])*(self.alpha/N_test)

            
        return H,h_test
    
    def PE_divergence(self,theta,data_ref=None,data_test=None):
    ### Function estimating the Pearson Divergence at the node level 

    ## Input
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v, n'_v, L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output 
    # score: numpy vector of dimension n_nodes; each element represents the Pearson Divergence at the node level.

        
        H,h_test=self.update_hs(data_ref,data_test)
        
        score=np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            score[i]=theta[i].dot(H[i]).dot(theta[i])
            score[i]*=-0.5
            score[i]+=theta[i].dot(h_test[i])
            score[i]-=0.5 
            
        return score
    
    def r_v(self,node_index,theta,data_node):
        
    ### Estimate likelihood-ratio function at node node_index

    ## Input
    # node_index: index of the node where the likelihood ratio is evaluated.
    # theta: estimated parameter.
    # data: data points used to evaluate the likelihood ratio at node v.
    
    ## Output
    # Likelihood ratio at node node_index evaluated at the points in data.

        
        aux_theta=1*theta[node_index]
        phi=self.kernel.k_V(transform_data(data_node))
        ratio=phi.dot( aux_theta)
 
        return ratio
    
    

class Pool():
### Class implementing the GRULSIF f-divergence estimation without the graph structure 
    def __init__(self,data_ref,data_test,threshold_coherence=0.3,alpha=0.1,tol=1e-2,k_cross_validation=5,verbose=False,kernel_type="Gaussian"):
    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
    #                      When the kernel is normal, this parameter should be between 0 and 1.
    #                      The closer it is to 1, the larger the dictionary and the slower the training.
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
    # tol: acceptable tolerance level in the estimation.
    # k_cross_validation: number of splits for cross-validation.   
    # verbose: whether or not to print intermediate steps.
    # kernel_type: the type of kernel used in the optimization method: "Gaussian" or "Laplace".
    
        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
    
        try:
            alpha = float(alpha)
            if not (0.0 <= alpha < 1):
                raise ValueError(f"Parameter alpha must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: alpha parameter should be a float")
            sys.exit(1)
            
        try:
            k_cross_validation=int(k_cross_validation)
            if not (1<k_cross_validation):
                raise ValueError(F"The number of splits should be bigger than 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  # Stops execution
            
        try:
            threshold_coherence=float(threshold_coherence)
            if not (0.0<threshold_coherence<1):
                raise ValueError(F"The threshold coherence must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError as e:
            print("Error: The threshold coherence parameter should be a float")
            sys.exit(1)  
            
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)  
            
        try: 
            if kernel_type not in ["Gaussian","Laplace"]: 
                raise ValueError(F"Kernel sould be or Gausian or Laplace")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  

   
        
        self.verbose=verbose
        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.alpha=alpha
        self.n_nodes=len(data_ref)
        self.N=sum([len(d) for d in data_ref])+sum([len(d) for d in data_test])
        self.sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])
        self.kernel,self.gamma=CROSS_validation_Pool(k_cross_validation,self.data_ref,self.data_test,
                                                        self.alpha,tol=tol,threshold_coherence=threshold_coherence,sigma_list=self.sigma_list,verbose=self.verbose,kernel_type=kernel_type)
       
        self.L=len(self.kernel.dictionary)
   
    def fit(self,data_ref=None,data_test=None,tol=1e-3,verbose=False): 
    ### Function estimating the theta parameter associated with the likelihood ratios 
    ### for a given set of observations coming from p_v(.) and q_v(.) 

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # tol: acceptable tolerance level in the estimation.
    # verbose: whether or not to print the iterations.

    ## Output
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)  
        
        h_s=self.update_hs(data_ref,data_test)
        theta_ini=1e-6*np.ones((self.n_nodes,self.kernel.n),dtype=np.float32)
        learning_rates=np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            h_s[0][i]/=self.n_nodes
            h_s[0][i]+=1e-6*np.eye(self.kernel.n)
            eta_i, _ = eigsh(h_s[0][i],k=1,ncv=np.min((500,self.kernel.n)))        
            learning_rates[i]=1*eta_i
            
        theta=optimize_Pool(theta_ini,h_s[0],h_s[1],learning_rates,self.alpha,self.gamma,tol=tol,verbose=verbose)
        
        return theta
        
    def update_hs(self,data_ref=None,data_test=None):
    ####This function estimates the variables H and h_test for each parameter in the estimation of GRULSIF

    ## Input

    # data_ref: data points representing the distribution p_v(.).
    # data_test: data points representing the distribution q_v(.).

    ## Output
    # H: numpy matrix of dimension n_nodes x L x L; the first coordinate indicates a matrix of 
    #    (1 - alpha) * K() K()^T + alpha * K() K()^T over observations from the data points coming from p_v and q_v.
    # h_test: numpy matrix of dimension n_nodes x L; each row is a vector with means of K() 
    #         over observations from the distribution q_v(.).

        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
                
        phi_test=[self.kernel.k_V(data_test[i]) for i in range(self.n_nodes)]
        phi_ref=[self.kernel.k_V(data_ref[i]) for i in range(self.n_nodes)]
    
    ############# Estimate H
        h_test =np.zeros((self.n_nodes,self.kernel.n),dtype=np.float32)
        H = np.zeros((self.n_nodes,self.kernel.n,self.kernel.n),dtype=np.float32) 
        for i in range(self.n_nodes):
            N_ref=len(phi_ref[i])
            N_test=len(phi_test[i])
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H[i]*=(1-self.alpha)/N_ref
            H[i]+=np.einsum('ji,j...',  phi_test[i],phi_test[i])*(self.alpha/N_test)
  
        return H,h_test
    
    def PE_divergence(self,theta,data_ref=None,data_test=None):
    ### Function estimating the Pearson Divergence at the node level 

    ## Input
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v, n'_v, L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output 
    # score: numpy vector of dimension n_nodes; each element represents the Pearson Divergence at the node level.
    
        H,h_test=self.update_hs(data_ref,data_test)
        score=np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            score[i]=theta[i].dot(H[i]).dot(theta[i])
            score[i]*=-0.5
            score[i]+=theta[i].dot(h_test[i])
            score[i]-=0.5 
        return score
            
    def r_v(self,node_index,theta,data_node):
   ### Estimate likelihood-ratio function at node node_index

   ## Input
   # node_index: index of the node where the likelihood ratio is evaluated.
   # theta: estimated parameter.
   # data: data points used to evaluate the likelihood ratio at node v.
   
   ## Output
   # Likelihood ratio at node node_index evaluated at the points in data.

   
        aux_theta=1*theta[node_index]
        phi=self.kernel.k_V(transform_data(data_node))
        ratio=phi.dot(aux_theta)
 
        return ratio    

    
    
 
    
    
    
    