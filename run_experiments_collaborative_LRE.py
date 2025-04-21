# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  run_experiments_collaborative_LRE
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-26              
# This version:     2025-02-26
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to replicate the experiments presented in the paper. 
#               It reproduces results similar to those appearing in the paper.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: pickle, numpy, argparse, Experiments.aux_synthetic_experiments_LRE, Evaluation.evaluation_metrics
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: GRULSIF, RULSIF, ULSIF, KLIEP, Pool, likelihood-ratio estimation, collaborative likelihood-ratio estimation.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
from Experiments.aux_synthetic_experiments_LRE import *
from Evaluation.evaluation_metrics import *
import pickle
import numpy as np

def main(results_directory,model,experiment,n_nodes,N_ref,N_test,alpha=0.1,threshold_coherence=0.3,N_runs=10,time_it=False): 
    
    ### This function estimates the LRE for each of the nodes for all the methods being analyzed. 

    #### Input
    # results_directory: the name of the directory where the results will be stored.
    # model: the model to be used; available options are "grulsif", "pool", "rulsif", "ulsif", and "kliep".
    # n_nodes: int, the number of nodes to be used.
    # N_ref: the number of observations coming from p_v; the same number will be used for all nodes.
    # N_test: the number of observations coming from q_v; the same number will be used for all nodes.
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
    # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
    #                      When the kernel is normal, this parameter should be between 0 and 1.
    #                      The closer it is to 1, the larger the dictionary and the slower the training.
    # N_runs: the number of random instances to evaluate the model.
    # time_it: whether or not to save the computational time of the algorithms.
    
    try: 
        if model not in ["grulsif","pool","rulsif","ulsif","kliep"]: 
            raise ValueError(F"Experiment should be grulsif, pool, rulsif, ulsif, kliep")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1) 

    
    if model=="grulsif":
        
        if time_it:
            
            list_times=[]
            list_model_parameters=[]
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                print(i)
                LREmodel_,model_theta,affected_nodes,t=aux_run_GRULSIF_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence,time_it=time_it)  
                model_parameters={"kernel":LREmodel_.kernel,"lamb":LREmodel_.lamb,
                          "gamma":LREmodel_.gamma,"G":LREmodel_.W}
                time_fit={"time_fit":t}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}                
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_times.append(time_fit)
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
            
            file_name=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            file_name_time=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}_time"
            
        else:
            list_model_parameters=[]
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                print(i)
                LREmodel_,model_theta,affected_nodes=aux_run_GRULSIF_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence)  
                model_parameters={"kernel":LREmodel_.kernel,"lamb":LREmodel_.lamb,
                          "gamma":LREmodel_.gamma,"G":LREmodel_.W}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)

                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
            
            file_name=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            
        
        
    elif model=="pool":
        
        if time_it:
            list_times=[]
            list_model_parameters=[]
            list_results_LRE=[]
            list_errors=[]

            
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes,t=aux_run_Pool_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence,time_it=time_it)  
                model_parameters={"kernel":LREmodel_.kernel,"gamma":LREmodel_.gamma}
                time_fit={"time_fit":t}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_times.append(time_fit)
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
              
            file_name=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            file_name_time=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}_time"
            
        else:
            list_model_parameters=[]
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes=aux_run_Pool_experiments(experiment,n_nodes,N_ref,N_test,alpha,threshold_coherence)  
                model_parameters={"kernel":LREmodel_.kernel,"gamma":LREmodel_.gamma}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)
                list_errors.append(errors_)

            file_name=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            
      
    elif model=="rulsif": 
        
        if time_it:
            list_times=[]
            list_model_parameters=[]   
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes,t=aux_run_Rulsif_experiments(experiment,n_nodes,N_ref,N_test,alpha,time_it=time_it) 
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.RULSIF_models[j].kernel for j in range(n_nodes)]
                gammas_=[LREmodel_.RULSIF_models[j].gamma for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_,"gamma":gammas_} 
                time_fit={"time_fit":t}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_times.append(time_fit)
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
              
            
            file_name=results_directory+"/"+f"Rulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            file_name_time=results_directory+"/"+f"Rulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}_time"
            
            
        else:            
            list_model_parameters=[]  
            list_results_LRE=[]
            list_errors=[]
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes=aux_run_Rulsif_experiments(experiment,n_nodes,N_ref,N_test,alpha) 
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.RULSIF_models[j].kernel for j in range(n_nodes)]
                gammas_=[LREmodel_.RULSIF_models[j].gamma for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_,"gamma":gammas_} 
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
            
            file_name=results_directory+"/"+f"Rulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N_ref}"
            
            
        
    elif model=="ulsif": 
        
        if time_it:
            list_times=[]
            list_model_parameters=[]  
            list_results_LRE=[]
            list_errors=[]
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes,t=aux_run_Ulsif_experiments(experiment,n_nodes,N_ref,N_test,time_it=True)        
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.ULSIF_models[j].kernel for j in range(n_nodes)]
                gammas_=[LREmodel_.ULSIF_models[j].gamma for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_,"gamma":gammas_} 
                time_fit={"time_fit":t}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_times.append(time_fit)
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
            
            file_name=results_directory+"/"+f"Ulsif_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}"
            file_name_time=results_directory+"/"+f"Ulsif_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}_time"
            
        else: 
            list_model_parameters=[] 
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                print(i)
                LREmodel_,model_theta,affected_nodes=aux_run_Ulsif_experiments(experiment,n_nodes,N_ref,N_test) 
        
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.ULSIF_models[j].kernel for j in range(n_nodes)]
                gammas_=[LREmodel_.ULSIF_models[j].gamma for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_,"gamma":gammas_} 
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)

                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)
            
            
            file_name=results_directory+"/"+f"Ulsif_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}"
            
            
    
    elif model=="kliep": 
    
        if time_it:
            list_times=[]
            list_model_parameters=[]  
            list_results_LRE=[]
            list_errors=[]
             
            for i in range(N_runs):
                LREmodel_,model_theta,affected_nodes,t=aux_run_Kliep_experiments(experiment,n_nodes,N_ref,N_test,time_it=True)        
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.KLIEP_models[j].kernel for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_} 
                time_fit={"time_fit":t}
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_times.append(time_fit)
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)

            
            file_name=results_directory+"/"+f"Kliep_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}"
            file_name_time=results_directory+"/"+f"Kliep_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}_time"
            
        else:
            list_model_parameters=[] 
            list_results_LRE=[]
            list_errors=[]
            
            for i in range(N_runs):
                print(i)
                LREmodel_,model_theta,affected_nodes=aux_run_Kliep_experiments(experiment,n_nodes,N_ref,N_test)
        
                n_nodes=len(model_theta)
                kernels_=[LREmodel_.KLIEP_models[j].kernel for j in range(n_nodes)]
        
                model_parameters={"kernel":kernels_} 
                results_LRE={"theta":model_theta,"LREmodel_":model_parameters,"affected_nodes":affected_nodes}
                errors_=evaluate_models(experiment,n_nodes,alpha,model_theta,LREmodel_)
                                
                list_model_parameters.append(model_parameters)
                list_results_LRE.append(results_LRE)  
                list_errors.append(errors_)


            file_name=results_directory+"/"+f"Kliep_experiment_{experiment}_n_nodes_{n_nodes}_N_{N_ref}"
            
    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(list_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(list_results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    if time_it:
        file_name_time=file_name_time.replace(".","")
        with open( file_name_time+".pickle", 'wb') as handle:
            pickle.dump(list_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--model",type=str) #### The name of the model to be run
    parser.add_argument("--experiment",type=str) #### The experiment to be run
    parser.add_argument("--n_nodes",type=int) #### The number of observations in the graph
    parser.add_argument("--alpha",default=0.0,type=float) #### the regularization parameter for the likelihood ratio 
    parser.add_argument("--threshold_coherence",default=None,type=float) #### parameter to the selection of the dictionary 
    parser.add_argument("--N_runs",default=1,type=int) #### number of repetition in the experiment
    parser.add_argument("--time_it",default=False,type=bool) #### seed to replicate the syntetic experiments
  
    args=parser.parse_args()
    
    results_directory=args.results_directory
    model=args.model
    experiment=args.experiment 
    n_nodes=args.n_nodes
    alpha=args.alpha
    threshold_coherence=args.threshold_coherence
    N_runs=args.N_runs
    time_it=args.time_it
    sample_size=np.array((50,100,250,500))
    for s in sample_size:
        main(results_directory,model,experiment,n_nodes,s,s,alpha,threshold_coherence,N_runs,time_it=time_it)
 


    
    