# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  run_experiments_collaborative_two_sample_test
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-27              
# This version:     2024-02-27
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to replicate the experiments presented in the paper. 
#               It replicates results similar to those presented in "Collaborative Non-Parametric Two-Sample Testing."
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: pickle, argparse, Evaluation.evaluation_metrics_ctst
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Collaborative TST, Pool, RULSIF, LSTT, MMD, C2ST, KLIEP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
from Evaluation.evaluation_metrics_ctst import *
import pickle


def main(results_directory,model,experiment,N,alpha=0.1,threshold_coherence=0.3):  
    
    ### This function computes the AFROC curve associated with each of the methods explored in the paper.

    #### Input:
    # results_directory: the name of the directory where the results will be stored.
    # model: the model to be used; available options are "GRULSIF", "POOL", "RULSIF", "ULSIF", and "KLIEP".
    # n_nodes: int, the number of nodes to be used.
    # N: the number of observations coming from p_v; the same number will be used for all nodes. (Same for q_v)
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
    # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
    #                      When the kernel is normal, this parameter should be between 0 and 1.
    #                      The closer it is to 1, the larger the dictionary and the slower the training.

    N_ref=N
    N_test=N
    try: 
        if model not in ["c2st","pool","rulsif","lstt","kliep","mmd_median","mmd_max"]: 
            raise ValueError(F"Experiment should be grulsif, pool, rulsif, lstt, kliep, mmd_median or mmd_max")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1) 
    
    if model=="c2st":
        FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative=AFROC_C2ST(experiment,N_ref,N_test,alpha,threshold_coherence,n_rounds=1000,seed=0)

        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}
        scores={"PE_Null_1":PE_scores_1_Null,"PE_Null_2":PE_scores_2_Null,
                "PE_Alternative_1":PE_scores_1_Alternative,"PE_Alternative_2":PE_scores_2_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        
    elif model=="pool":
        
        FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative=AFROC_POOL_tst(experiment,N_ref,N_test,alpha,threshold_coherence,n_rounds=1000,seed=0)

        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}
        scores={"PE_Null_1":PE_scores_1_Null,"PE_Null_2":PE_scores_2_Null,
                "PE_Alternative_1":PE_scores_1_Alternative,"PE_Alternative_2":PE_scores_2_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        
      
    elif model=="rulsif": 
        FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative=AFROC_RULSIF_tst(experiment,N_ref,N_test,alpha,n_rounds=1000,seed=0)

        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}

        scores={"PE_Null_1":PE_scores_1_Null,"PE_Null_2":PE_scores_2_Null,
                "PE_Alternative_1":PE_scores_1_Alternative,"PE_Alternative_2":PE_scores_2_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_Rulsif_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_Rulsif_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
        
        
    elif model=="lstt": 
        FWR,FPR,TPR,PE_scores_1_Null,PE_scores_2_Null,PE_scores_1_Alternative,PE_scores_2_Alternative=AFROC_LSTT(experiment,N_ref,N_test,n_rounds=1000,seed=0)

        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}
        scores={"PE_Null_1":PE_scores_1_Null,"PE_Null_2":PE_scores_2_Null,
                "PE_Alternative_1":PE_scores_1_Alternative,"PE_Alternative_2":PE_scores_2_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_lstt_experiment_{experiment}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_lstt_experiment_{experiment}_N_{N_ref}"
        
    
    elif model=="kliep": 
        FWR,FPR,TPR,KL_scores_1_Null,KL_scores_2_Null,KL_scores_1_Alternative,KL_scores_2_Alternative=AFROC_KLIEP_tst(experiment,N_ref,N_test,n_rounds=1000,seed=0)

        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}
        scores={"KL_Null_1":KL_scores_1_Null,"KL_Null_2":KL_scores_2_Null,
                "KL_Alternative_1":KL_scores_1_Alternative,"KL_Alternative_2":KL_scores_2_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_Kliep_experiment_{experiment}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_Kliep_experiment_{experiment}_N_{N_ref}"

    elif model=="mmd_median":
        
        FWR,FPR,TPR,MMD_scores_Null,MMD_scores_Alternative=AFROC_MMD_tst(experiment,N_ref,N_test,n_rounds=1000,seed=0)
        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR} 
        scores={"MMD_Null":MMD_scores_Null,
                "MMD_Alternative":MMD_scores_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_MMD_median_experiment_{experiment}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_MMD_median_experiment_{experiment}_N_{N_ref}"
        
    elif model=="mmd_max":
        
        FWR,FPR,TPR,MMD_scores_Null,MMD_scores_Alternative=AFROC_MMD_tst(experiment,N_ref,N_test,n_rounds=1000,seed=0,estimate_sigma=True)
        AFROC_parameters={"FWR":FWR,"FPR":FPR,"TPR":TPR}
        scores={"MMD_Null":MMD_scores_Null,
                "MMD_Alternative":MMD_scores_Alternative}                 
        
        file_name_AFROC=results_directory+"/"+f"AFROC_MMD_max_experiment_{experiment}_N_{N_ref}"
        file_name_scores=results_directory+"/"+f"scores_MMD_max_experiment_{experiment}_N_{N_ref}"
        
        
    file_name_AFROC=file_name_AFROC.replace(".","")
    file_name_scores=file_name_scores.replace(".","")
    
    
    with open( file_name_AFROC+".pickle", 'wb') as handle:
        pickle.dump(AFROC_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
    with open( file_name_scores+".pickle", 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--model",type=str) #### The name of the model to be run
    parser.add_argument("--experiment",type=str) #### The experiment to be runh
    parser.add_argument("--N",type=int)  ############## The number of observations generated by p_v (q_v)
   # parser.add_argument("--N_test",type=int) #### The number of observations generated by q_v
    parser.add_argument("--alpha",default=0.0,type=float) #### the regularization parameter for the likelihood ratio 
    parser.add_argument("--threshold_coherence",default=None,type=float) #### parameter to the selection of the dictionary 
  
    args=parser.parse_args()
    
    results_directory=args.results_directory
    model=args.model
    experiment=args.experiment 
    N=args.N
    alpha=args.alpha
    threshold_coherence=args.threshold_coherence
    
    main(results_directory,model,experiment,N,alpha,threshold_coherence)
    
    
    