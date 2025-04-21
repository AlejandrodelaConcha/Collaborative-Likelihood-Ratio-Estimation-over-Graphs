# -----------------------------------------------------------------------------------------------------------------
# Title:  MMD_methods
# Author(s): Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-27              
# This version:     2025-02-27
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This script implements the MMD methods described in Gretton et al. (2012) and Sutherland et al. (2017).
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, numba, Models.aux_functions
# -----------------------------------------------------------------------------------------------------------------
# Keywords: MMD_implementation
# -----------------------------------------------------------------------------------------------------------------

from Models.aux_functions import *
import numpy as np
from numba import njit,jit
import copy


@jit(nopython=True)
def MMD(K,N_ref,N_test): 
    #### This function estimates the MMD score.

    ### Input:
    # K: Gram matrix.
    # N_ref: Number of observations coming from p_v.
    # N_test: Number of observations coming from q_v.

    ### Output:
    # mmd: estimated MMD score.

    
    k_x = K[:N_ref, :N_ref]
    np.fill_diagonal(k_x,0)
    k_y = K[N_ref:, N_ref:]
    np.fill_diagonal(k_y,0)
    k_xy = K[:N_ref, N_ref:]
    mmd = k_x.sum() / (N_ref * (N_ref - 1)) + k_y.sum() / (N_test * (N_test - 1)) - 2 * k_xy.sum() / (N_ref * N_test)
    return mmd
 
@jit(nopython=True)       
def VAR_MMD(K,N_ref,N_test): 
    ### This function estimates the variance of MMD based on Sutherland et al. (2017).

    ### Input:
    # K: Gram matrix.
    # N_ref: Number of observations coming from p_v.
    # N_test: Number of observations coming from q_v.

    ### Output:
    # var_mmd: estimated variance of MMD.

    
    Kxx = K[:N_ref, :N_ref]
    np.fill_diagonal(Kxx,0)
    Kxy = K[:N_ref, N_ref:]
    Kyx = K[N_ref:, :N_ref]
    Kyy = K[N_ref:, N_ref:]
    np.fill_diagonal(Kyy,0)
    H_column_sum = (
        np.sum(Kxx, axis=1)
        + np.sum(Kyy, axis=1)
        - np.sum(Kxy, axis=1)
        - np.sum(Kyx, axis=1)
    )
    var_mmd = (
        4 / N_ref ** 3 * np.sum(H_column_sum ** 2)
        - 4 / N_ref ** 4 * np.sum(H_column_sum) ** 2
        + 1e-6
    )
    return var_mmd
   

class MMD_nodes():
########## Class implementing MMD estimation for all nodes.
    def __init__(self,data_ref,data_test,estimate_sigma=False):     
    
    ## Input:
    # data_ref: data points representing the distribution p_v(.).
    # data_test: data points representing the distribution q_v'(.).
    # estimate_sigma: Boolean variable indicating whether to estimate the width parameter via the standardized MMD.

        
        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.n_nodes=len(self.data_ref)
        self.sigmas=self.initializalize_kernel(estimate_sigma) 

    def initializalize_kernel(self,estimate_sigma=False):
    
    ## Input:
    # estimate_sigma: Boolean variable indicating whether to estimate the width parameter via the standardized MMD.

    
        sigmas=np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if self.data_test[i].shape[0]>100:
                index_test=np.random.choice(len(self.data_test[i]),replace=False,size=100)               
                aux_sigmas=calc_dist(self.data_ref[i][index_test],self.data_test[i][index_test],sqrt=True)
                aux_sigmas[np.isnan(aux_sigmas)]=0.0
                sigmas[i]=np.sqrt((np.median( aux_sigmas[np.triu_indices(100,1)])**2)/2)
            else:
                n=len(self.data_ref[i])
                aux_sigmas=calc_dist(self.data_ref[i],self.data_test[i],sqrt=True)
                aux_sigmas[np.isnan( aux_sigmas)]=0.0
                sigmas[i]=np.sqrt((np.median(aux_sigmas[np.triu_indices(n,1)])**2)/2)
            
            if sigmas[i]==0:
                sigmas[i]=1e-6
                
            if estimate_sigma: 
                
                N_ref=int(np.min((len(self.data_test[i]),len(self.data_ref[i])))/2)
                N_test=N_ref
                index_test=np.random.choice(int(2*N_ref),replace=False,size=N_ref) 
                aux_data=np.vstack((self.data_ref[i][index_test],self.data_test[i][index_test]))
                sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])*sigmas[i]
                score=np.zeros(len(sigma_list))
                for i in range(len(sigma_list)):
                    K=product_gaussian(sigma_list[i],aux_data,aux_data)
                    mmd_s=MMD(K,N_ref,N_test)
                    var_mmd_s=VAR_MMD(K,N_ref,N_test)
                    score[i]=mmd_s/np.sqrt(var_mmd_s)
                sigmas[i]=sigma_list[np.argmax(score)]
        return sigmas

           
    def get_MMD(self,data_ref=None,data_test=None):
    ####### Function estimating the MMD score at each of the nodes.
    ### Input:
    # data_ref: data points representing the distribution p_v(.).
    # data_test: data points representing the distribution q_v(.).
    ### Output:
    # score: MMD scores at the node level.
      
        if  data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=[transform_data(d) for d in data_test]
        else:
            data_test=self.data_test

        N_ref=data_ref[0].shape[0]
        N_test=data_test[0].shape[0]       
        n_nodes=len(data_ref)
        scores=np.zeros(n_nodes)
        
        for i in range(n_nodes):
            aux_data=np.vstack((data_ref[i],data_test[i]))
            K=product_gaussian(self.sigmas[i],aux_data,aux_data)
            scores[i]=MMD(K,N_ref,N_test)
        
        return scores

    
        
   
    
   