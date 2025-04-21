# -----------------------------------------------------------------------------------------------------------------
# Title:  aux_synthectic_experiments_LRE
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2024-02-26             
# This version:     2024-02-26
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to evaluate the models presented in the paper over the experiments 1A,1B,2A,2B an 2C
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, scipy, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GRULSIF, RULSIF, ULSIF, KLIEP, Pool, likelihood-ratio estimation, collaborative likelihood-ratio estimation.
# -----------------------------------------------------------------------------------------------------------------


from Experiments.experiments_LRE import generate_experiment_1_LRE,generate_experiment_2_LRE,generate_experiment_3_LRE,generate_experiment_4_LRE
from Models.likelihood_ratio_collaborative import *
from Models.likelihood_ratio_univariate import *
import time
import sys

def aux_run_GRULSIF_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence,time_it=False):
### This function fits the GRULSIF likelihood estimator over a given experiment. 
### In this implementation, we assume all nodes have the same number of observations, 
### which is why N_ref and N_test are integers.

## Input: 
# experiment: ["1A","1B","2A","2B","2C"] synthetic experiment to be evaluated .
# n_nodes: number of nodes in the graph.
# N_ref: number of observations per node coming from p_v.
# N_test: number of observations per node coming from q_v.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
#                      When the kernel is normal, this parameter should be between 0 and 1.
#                      The closer it is to 1, the larger the dictionary and the slower the training.
# time_it: whether or not to save the computational time of the algorithms.

    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)        
 

    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)    
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    elif experiment=="2B":
        d=2
        G,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
    elif experiment=="2C":
        d=10
        G,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)         
        
    grulsif_=GRULSIF(G.W,data_ref,data_test,threshold_coherence,alpha=alpha,verbose=False)
       
    if time_it:
        start=time.time()
        grulsif_theta=grulsif_.fit(data_ref,data_test)
        end=time.time()
        t=end-start 
        return grulsif_,grulsif_theta,affected_nodes,t
    
    else:
        grulsif_theta=grulsif_.fit(data_ref,data_test)
        return grulsif_,grulsif_theta,affected_nodes
        
def aux_run_Pool_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence,time_it=False):
### This function fits the Pool likelihood estimator over a given experiment. 
### In this implementation, we assume all nodes have the same number of observations, 
### which is why N_ref and N_test are integers.

## Input: 
# experiment: ["1A","1B","2A","2B","2C"] synthetic experiment to be evaluated .
# n_nodes: number of nodes in the graph.
# N_ref: number of observations per node coming from p_v.
# N_test: number of observations per node coming from q_v.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
#                      When the kernel is normal, this parameter should be between 0 and 1.
#                      The closer it is to 1, the larger the dictionary and the slower the training.
# time_it: whether or not to save the computational time of the algorithms.    


    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)        
 
    
    if experiment=="1A":
        _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)    
    elif experiment=="1B":
        _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)
    elif experiment=="2A":
        _,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    elif experiment=="2B":
        d=2
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
    elif experiment=="2C":
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
          
     
    pool_=Pool(data_ref,data_test,threshold_coherence,alpha=alpha,verbose=False)
    pool_theta=pool_.fit(data_ref,data_test)
    
    if time_it:
        start=time.time()
        pool_theta=pool_.fit(data_ref,data_test)
        end=time.time()
        t=end-start 
        return pool_,pool_theta,affected_nodes,t  
         
    else:
        pool_theta=pool_.fit(data_ref,data_test)
        return pool_,pool_theta,affected_nodes


def aux_run_Rulsif_experiments(experiment,n_nodes,N_ref,N_test,alpha,time_it=False):
### This function fits the RULSIF likelihood estimator over a given experiment. 
### In this implementation, we assume all nodes have the same number of observations, 
### which is why N_ref and N_test are integers.

## Input: 
# experiment: ["1A","1B","2A","2B","2C"] synthetic experiment to be evaluated .
# n_nodes: number of nodes in the graph.
# N_ref: number of observations per node coming from p_v.
# N_test: number of observations per node coming from q_v.
# alpha: regularization parameter associated with the upper bound of the likelihood ratio.
# time_it: whether or not to save the computational time of the algorithms.

    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)        

    if experiment=="1A":
        _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)    
    elif experiment=="1B":
        _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)       
    elif experiment=="2A":
        _,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    elif experiment=="2B":
        d=2
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
    elif experiment=="2C":
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
            
    rulsif_=RULSIF_nodes(data_ref,data_test,alpha=alpha)
    
    if time_it:
        start=time.time()
        rulsif_theta=rulsif_.fit(data_ref,data_test)
        end=time.time()
        t=end-start 
        return rulsif_,rulsif_theta,affected_nodes,t
    
    else:
        rulsif_theta=rulsif_.fit(data_ref,data_test)
        return rulsif_,rulsif_theta,affected_nodes
        

       
def aux_run_Ulsif_experiments(experiment,n_nodes,N_ref,N_test,seed=0,time_it=False):
### This function fits the ULSIF likelihood estimator over a given experiment. 
### In this implementation, we assume all nodes have the same number of observations, 
### which is why N_ref and N_test are integers.

## Input: 
# experiment: ["1A","1B","2A","2B","2C"] synthetic experiment to be evaluated .
# n_nodes: number of nodes in the graph.
# N_ref: number of observations per node coming from p_v.
# N_test: number of observations per node coming from q_v.
# time_it: whether or not to save the computational time of the algorithms.

    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)        

    
    if experiment=="1A":
        _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)    
    elif experiment=="1B":
        _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)
    elif experiment=="2A":
        _,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    elif experiment=="2B":
        d=2
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
    elif experiment=="2C":
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
       

        
    ulsif_=ULSIF_nodes(data_ref,data_test)
    
    if time_it:
        start=time.time()
        ulsif_theta=ulsif_.fit(data_ref,data_test)
        end=time.time()
        t=end-start 
        return ulsif_,ulsif_theta,affected_nodes,t
        
    
    else:
        ulsif_theta=ulsif_.fit(data_ref,data_test)
        return ulsif_,ulsif_theta,affected_nodes


def aux_run_Kliep_experiments(experiment,n_nodes,N_ref,N_test,seed=0,time_it=False):
### This function fits the KLIEP likelihood estimator over a given experiment. 
### In this implementation, we assume all nodes have the same number of observations, 
### which is why N_ref and N_test are integers.

## Input: 
# experiment: ["1A","1B","2A","2B","2C"] synthetic experiment to be evaluated .
# n_nodes: number of nodes in the graph.
# N_ref: number of observations per node coming from p_v.
# N_test: number of observations per node coming from q_v.
# time_it: whether or not to save the computational time of the algorithms.

    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)        
 
    if experiment=="1A":
        _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)    
    elif experiment=="1B":
        _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01)
    elif experiment=="2A":
        _,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    elif experiment=="2B":
        d=2
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
    elif experiment=="2C":
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
               
    kliep_=KLIEP_nodes(data_ref,data_test)
    
    if time_it:
        start=time.time()
        kliep_theta=kliep_.fit(data_ref,data_test)
        end=time.time()
        t=end-start 
        return kliep_,kliep_theta,affected_nodes,t

    else:
        kliep_theta=kliep_.fit(data_ref,data_test)
        return kliep_,kliep_theta,affected_nodes

              
     
        
    
        
        
        
        
        
        