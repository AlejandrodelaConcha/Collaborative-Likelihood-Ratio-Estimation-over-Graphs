# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  evaluation_metrics_ctst
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-27              
# This version:     2025-02-27
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to replicate the AFROC curves produced in the paper 
#               "Collaborative Non-Parametric Two-Sample Testing."
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy, sklearn, Experiments, Models
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: AFROC, C2ST, Pool, ULSIF, RULSIF, KLIEP, MMD
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np 
from Experiments.experiments_two_sample_test import generate_experiment_1,generate_experiment_2,generate_experiment_3,generate_experiment_4
from Models.likelihood_ratio_collaborative import *
from Models.likelihood_ratio_univariate import *
from Models.MMD_methods import *
from Models.aux_functions import *
from sklearn import metrics


def AFROC_C2ST(experiment,N_ref,N_test,alpha,threshold_coherence,n_rounds=1000,seed=0):
    ######## Function to estimate the AFROC curve for C2ST.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # alpha: regularization parameter of the likelihood ratio.
    # threshold_coherence: parameter related to the dictionary.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # PE_scores_1_NULL, PE_scores_2_NULL: The estimated Pearson divergence scores when for all null hypotheses \( p_v = q_v \).
    # PE_scores_1_Alternative, PE_scores_2_Alternative: The estimated Pearson divergence scores when the scenarios presented in the paper are satisfied.


    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
        
    W=G.W.tocoo()   
    n_nodes=len(data_ref)
    grulsif_1=GRULSIF(G.W,data_ref,data_test,threshold_coherence,alpha=alpha,verbose=False)
    theta_1=grulsif_1.fit()
    scores_1=grulsif_1.PE_divergence(theta_1)
    
    grulsif_2=GRULSIF(G.W,data_test,data_ref,threshold_coherence,alpha=alpha,verbose=False)
    theta_2=grulsif_2.fit()
    scores_2=grulsif_2.PE_divergence(theta_2)
    
    PE_scores_1_Null=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Null=np.zeros((n_rounds,n_nodes))
    
    PE_scores_1_Alternative=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Alternative=np.zeros((n_rounds,n_nodes))
      
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0,H_null=True)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0,H_null=True)
        
        theta_1=grulsif_1.fit(data_ref,data_test)
        PE_scores_1_Null[i]=grulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=grulsif_2.fit(data_test,data_ref)
        PE_scores_2_Null[i]=grulsif_2.PE_divergence(theta_2,data_test,data_ref)
  
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)        
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
        
            
        theta_1=grulsif_1.fit(data_ref,data_test)
        PE_scores_1_Alternative[i]=grulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=grulsif_2.fit(data_test,data_ref)
        PE_scores_2_Alternative[i]=grulsif_2.PE_divergence(theta_2,data_test,data_ref)

    max_value=np.max((np.max(PE_scores_1_Alternative),np.max(PE_scores_2_Alternative)))
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(PE_scores_1_Null>=thresholds[i],axis=1)
        aux_FWR+=np.sum(PE_scores_2_Null>=thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(PE_scores_1_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_FPS=set(aux_indices.flatten())
               aux_indices=np.where(PE_scores_2_Alternative[j]>=thresholds[i])[0]
               aux_TPS=aux_TPS.union(set_affected_nodes.intersection(set(aux_indices.flatten())))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes)    
               aux_FPS=aux_FPS.union(set(aux_indices.flatten()))
               false_positives=len(aux_FPS-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
               
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
    
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR))    
    
    index_FWR=np.where(FWR<0.05)[0]
    index_FPR=np.where(FPR<0.05)[0]

        
    return FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative



def AFROC_POOL_tst(experiment,N_ref,N_test,alpha,threshold_coherence,n_rounds=1000,seed=0):
    ######## Function to estimate the AFROC curve for Pool.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # alpha: regularization parameter of the likelihood ratio.
    # threshold_coherence: parameter related to the dictionary.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # PE_scores_1_NULL, PE_scores_2_NULL: The estimated Pearson divergence scores when for all null hypotheses \( p_v = q_v \).
    # PE_scores_1_Alternative, PE_scores_2_Alternative: The estimated Pearson divergence scores when the scenarios presented in the paper are satisfied.


    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
    
   
    n_nodes=len(data_ref)        
    W=G.W.tocoo()
    pool_1=Pool(data_ref,data_test,threshold_coherence,alpha=alpha,verbose=False)
    theta_1=pool_1.fit()
    scores_1=pool_1.PE_divergence(theta_1)

    pool_2=Pool(data_test,data_ref,threshold_coherence,alpha=alpha,verbose=False)
    theta_2=pool_2.fit()
    scores_2=pool_2.PE_divergence(theta_2)

    PE_scores_1_Null=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Null=np.zeros((n_rounds,n_nodes))
    
    PE_scores_1_Alternative=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Alternative=np.zeros((n_rounds,n_nodes))

      
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0,H_null=True)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0,H_null=True)
        
        theta_1=pool_1.fit(data_ref,data_test)
        PE_scores_1_Null[i]=pool_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=pool_2.fit(data_test,data_ref)
        PE_scores_2_Null[i]=pool_2.PE_divergence(theta_2,data_test,data_ref)
  
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
       
        
        theta_1=pool_1.fit(data_ref,data_test)
        PE_scores_1_Alternative[i]=pool_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=pool_2.fit(data_test,data_ref)
        PE_scores_2_Alternative[i]=pool_2.PE_divergence(theta_2,data_test,data_ref)
  
    max_value=np.max((np.max(PE_scores_1_Alternative),np.max(PE_scores_2_Alternative)))
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(PE_scores_1_Null>thresholds[i],axis=1)
        aux_FWR+=np.sum(PE_scores_2_Null>thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(PE_scores_1_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_FPS=set(aux_indices.flatten())
               aux_indices=np.where(PE_scores_2_Alternative[j]>=thresholds[i])[0]
               aux_TPS=aux_TPS.union(set_affected_nodes.intersection(set(aux_indices.flatten())))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes)    
               aux_FPS=aux_FPS.union(set(aux_indices.flatten()))
               false_positives=len(aux_FPS-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
               
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
    
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR)) 
 
    
    return FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative


def AFROC_RULSIF_tst(experiment,N_ref,N_test,alpha,n_rounds=1000,seed=0):
    ######## Function to estimate the AFROC curve for RULSIF.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # alpha: regularization parameter of the likelihood ratio.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # PE_scores_1_NULL, PE_scores_2_NULL: The estimated Pearson divergence scores when for all null hypotheses \( p_v = q_v \).
    # PE_scores_1_Alternative, PE_scores_2_Alternative: The estimated Pearson divergence scores when the scenarios presented in the paper are satisfied.

    
    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
     
    n_nodes=len(data_ref)
    rulsif_1=RULSIF_nodes(data_ref,data_test,alpha=alpha)
    theta_1=rulsif_1.fit()
    scores_1=rulsif_1.PE_divergence(theta_1)

    rulsif_2=RULSIF_nodes(data_test,data_ref,alpha=alpha)
    theta_2=rulsif_2.fit()
    scores_2=rulsif_2.PE_divergence(theta_2)

    PE_scores_1_Null=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Null=np.zeros((n_rounds,n_nodes))

    PE_scores_1_Alternative=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Alternative=np.zeros((n_rounds,n_nodes))

    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0,H_null=True)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0,H_null=True)
        
        theta_1=rulsif_1.fit(data_ref,data_test)
        PE_scores_1_Null[i]=rulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=rulsif_2.fit(data_test,data_ref)
        PE_scores_2_Null[i]=rulsif_2.PE_divergence(theta_2,data_test,data_ref)
  
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
       
        
        theta_1=rulsif_1.fit(data_ref,data_test)
        PE_scores_1_Alternative[i]=rulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=rulsif_2.fit(data_test,data_ref)
        PE_scores_2_Alternative[i]=rulsif_2.PE_divergence(theta_2,data_test,data_ref)
  
    max_value=np.max((np.max(PE_scores_1_Alternative),np.max(PE_scores_2_Alternative)))
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(PE_scores_1_Null>=thresholds[i],axis=1)
        aux_FWR+=np.sum(PE_scores_2_Null>=thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(PE_scores_1_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_FPS=set(aux_indices.flatten())
               aux_indices=np.where(PE_scores_2_Alternative[j]>=thresholds[i])[0]
               aux_TPS=aux_TPS.union(set_affected_nodes.intersection(set(aux_indices.flatten())))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes)    
               aux_FPS=aux_FPS.union(set(aux_indices.flatten()))
               false_positives=len(aux_FPS-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
               
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
        
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR))
     
    return FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative


def AFROC_LSTT(experiment,N_ref,N_test,n_rounds=1000,seed=0):
    ######## Function to estimate the AFROC curve for LSTT.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # PE_scores_1_NULL, PE_scores_2_NULL: The estimated Pearson divergence scores when for all null hypotheses \( p_v = q_v \).
    # PE_scores_1_Alternative, PE_scores_2_Alternative: The estimated Pearson divergence scores when the scenarios presented in the paper are satisfied.

        
    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
    
    n_nodes=len(data_ref)    
    ulsif_1=ULSIF_nodes(data_ref,data_test)
    theta_1=ulsif_1.fit()
    scores_1=ulsif_1.PE_divergence(theta_1)

    ulsif_2=ULSIF_nodes(data_test,data_ref)
    theta_2=ulsif_2.fit()
    scores_2=ulsif_2.PE_divergence(theta_2)

    PE_scores_1_Null=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Null=np.zeros((n_rounds,n_nodes))

    PE_scores_1_Alternative=np.zeros((n_rounds,n_nodes))
    PE_scores_2_Alternative=np.zeros((n_rounds,n_nodes))

    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0,H_null=True)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0,H_null=True)
        
        theta_1=ulsif_1.fit(data_ref,data_test)
        PE_scores_1_Null[i]=ulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=ulsif_2.fit(data_test,data_ref)
        PE_scores_2_Null[i]=ulsif_2.PE_divergence(theta_2,data_test,data_ref)
  
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
       
        theta_1=ulsif_1.fit(data_ref,data_test)
        PE_scores_1_Alternative[i]=ulsif_1.PE_divergence(theta_1,data_ref,data_test)
        theta_2=ulsif_2.fit(data_test,data_ref)
        PE_scores_2_Alternative[i]=ulsif_2.PE_divergence(theta_2,data_test,data_ref)

    max_value=np.max((np.max(PE_scores_1_Alternative),np.max(PE_scores_2_Alternative)))
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(PE_scores_1_Null>=thresholds[i],axis=1)
        aux_FWR+=np.sum(PE_scores_2_Null>=thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(PE_scores_1_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_FPS=set(aux_indices.flatten())
               aux_indices=np.where(PE_scores_2_Alternative[j]>=thresholds[i])[0]
               aux_TPS=aux_TPS.union(set_affected_nodes.intersection(set(aux_indices.flatten())))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes) 
               aux_FPS=aux_FPS.union(set(aux_indices.flatten()))
               false_positives=len(aux_FPS-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
               
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR)) 

    return FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative


def AFROC_KLIEP_tst(experiment,N_ref,N_test,n_rounds=1000,seed=0):
    ######## Function to estimate the AFROC curve for KLIEP.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # KL_scores_1_NULL, KL_scores_2_NULL: The estimated Kullback-Leibler divergence scores when for all null hypotheses \( p_v = q_v \).
    # KL_scores_1_Alternative, KL_scores_2_Alternative: The estimated Kullback-Leibler divergence scores when the scenarios presented in the paper are satisfied.

        
    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
      
    n_nodes=len(data_ref)
    W=G.W.tocoo()
    kliep_1=KLIEP_nodes(data_ref,data_test)
    theta_1=kliep_1.fit()
    scores_1=kliep_1.KL_divergence(theta_1)

    kliep_2=KLIEP_nodes(data_test,data_ref)
    theta_2=kliep_2.fit()
    scores_2=kliep_2.KL_divergence(theta_2)

    KL_scores_1_Null=np.zeros((n_rounds,n_nodes))
    KL_scores_2_Null=np.zeros((n_rounds,n_nodes))
     
    KL_scores_1_Alternative=np.zeros((n_rounds,n_nodes))
    KL_scores_2_Alternative=np.zeros((n_rounds,n_nodes))

    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
           _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=seed)
        elif experiment=="2B":
           _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=seed)
           
        theta_1=kliep_1.fit(data_ref,data_test)
        KL_scores_1_Null[i]=kliep_1.KL_divergence(theta_1,data_test)
        theta_2=kliep_2.fit(data_test,data_ref)
        KL_scores_2_Null[i]=kliep_2.KL_divergence(theta_2,data_ref)
  
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)
       # elif experiment=="1C":
        #   _,data_ref,data_test,affected_nodes=generate_experiment_MNIST(data_directory,N_ref,N_test,seed=seed)  
        elif experiment=="2A":
           _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)
        elif experiment=="2B":
           _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
               
        theta_1=kliep_1.fit(data_ref,data_test)
        KL_scores_1_Alternative[i]=kliep_1.KL_divergence(theta_1,data_test)
        theta_2=kliep_2.fit(data_test,data_ref)
        KL_scores_2_Alternative[i]=kliep_2.KL_divergence(theta_2,data_ref)

    max_value=np.max((np.max(KL_scores_1_Alternative),np.max(KL_scores_2_Alternative)))
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(KL_scores_1_Null>=thresholds[i],axis=1)
        aux_FWR+=np.sum(KL_scores_2_Null>=thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(KL_scores_1_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_FPS=set(aux_indices.flatten())
               aux_indices=np.where(KL_scores_2_Alternative[j]>=thresholds[i])[0]
               aux_TPS=aux_TPS.union(set_affected_nodes.intersection(set(aux_indices.flatten())))
               aux_FPS=aux_FPS.union(set(aux_indices.flatten()))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes)    
               false_positives=len(aux_FPS-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
        
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR))

    return FWR,FPR,TPR,KL_scores_1_Null,KL_scores_2_Null,KL_scores_1_Alternative,KL_scores_2_Alternative


def AFROC_MMD_tst(experiment,N_ref,N_test,n_rounds=1000,seed=0,estimate_sigma=False):
    ######## Function to estimate the AFROC curve for MMD.

    ### Input:
    # experiment: scenario to replicate ("1A", "1B", "2A", "2B").
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # n_rounds: number of simulations to replicate.
    # seed: seed to fix the graph.
    # estimate_sigma: whether to estimate the sigma parameter for MMD.

    ### Output:
    # FWR: Fraction of iterations where there was at least one hypothesis to be rejected.
    # FPR: False Positive Rate, the fraction of nodes that were rejected but the null hypothesis was true (mean over the alternative hypotheses).
    # TPR: True Positive Rate, the fraction of nodes where the null hypothesis was correctly rejected.
    # MMD_scores_NULL: MMD scores when for all null hypotheses \( p_v = q_v \).
    # MMD_scores_Alternative: MMD scores when the scenarios presented in the paper are satisfied.
    if experiment=="1A":
        G,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="1B":
        G,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
    elif experiment=="2A":
        G,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
    elif experiment=="2B":
        G,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
      
    n_nodes=len(data_ref)
    W=G.W.tocoo()
    
    mmd_models=MMD_nodes(data_ref,data_test,estimate_sigma)
    score=mmd_models.get_MMD()
    threshold={"start":0,"step":np.max(score)/100}
    
    MMD_scores_Alternative=np.zeros((n_rounds,n_nodes))
    MMD_scores_Null=np.zeros((n_rounds,n_nodes))
 
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=True)
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0,H_null=True)  
        elif experiment=="2B":
            _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0,H_null=True)
        
        MMD_scores_Null[i]=mmd_models.get_MMD(data_ref,data_test)
 
    for i in range(n_rounds):
        if i%50==0:
            print(i)
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0) 
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0)  
        elif experiment=="2A":
             _,data_ref,data_test,affected_nodes=generate_experiment_3(N_ref,N_test,seed=0)  
        elif experiment=="2B":
             _,data_ref,data_test,affected_nodes=generate_experiment_4(N_ref,N_test,seed=0)
              
        MMD_scores_Alternative[i]=mmd_models.get_MMD(data_ref,data_test)
 
    max_value=np.max(MMD_scores_Alternative)
    thresholds=np.arange(0,max_value+1/100,step=max_value/100)
    
    FWR=np.zeros(len(thresholds))
    
    for i in range(len(thresholds)):
        aux_FWR=np.sum(MMD_scores_Null>=thresholds[i],axis=1)
        FWR[i]=np.sum(aux_FWR>=1)/n_rounds
    
    TPR=np.zeros(len(thresholds))
    FPR=np.zeros(len(thresholds))
    set_affected_nodes=set(affected_nodes.flatten())
    for i in range(len(thresholds)):
        aux_TPR=np.zeros(n_rounds)
        aux_FPR=np.zeros(n_rounds)
        for j in range(n_rounds):
               aux_indices=np.where(MMD_scores_Alternative[j]>=thresholds[i])[0]
               aux_TPS=set_affected_nodes.intersection(set(aux_indices.flatten()))
               aux_TPR[j]=len( aux_TPS)/len(set_affected_nodes)    
               false_positives=len(set(aux_indices.flatten())-set_affected_nodes)
               aux_FPR[j]=false_positives/(n_nodes-len(affected_nodes)) 
        TPR[i]=np.mean(aux_TPR)
        FPR[i]=np.mean(aux_FPR)
    TPR=np.hstack((1.0,TPR))
    FPR=np.hstack((1.0,FPR))
     
    return FWR,FPR,TPR,MMD_scores_Null,MMD_scores_Alternative

 
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        