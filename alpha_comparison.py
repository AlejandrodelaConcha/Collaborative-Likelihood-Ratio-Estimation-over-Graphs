# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title: alpha_comparison
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-26              
# This version:     2025-02-26
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal is to generate the plots appearing in the paper, where different values of alpha are used.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: argparse, pickle, numpy, matplotlib, Experiments, Evaluation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Pearson divergence, L2 convergence, Kullback-Leibler divergence
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Comments: To run this code, it is necessary to have executed `run_experiments_collaborative_LRE` 
#           in order to produce all the relevant files.

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Experiments.experiments_LRE import *
from Evaluation.evaluation_metrics import *


def plot_convergence_L2_LRE_alphas(results_directory,experiment,sample_sizes):
    
    ############ This function generates the L2-convergence plots presented in the paper. 

    ### Input
    # results_directory: the directory where the plots will be saved.
    # experiment: the experiment being studied: '1A', '1B', '2A', '2B', '2C'.
    # sample_sizes: sample sizes that were explored in the experiments.
    
    ############### Open files 
    
    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)   
    
    n_nodes=100
    
    mean_grulsif_errors_001=np.zeros(len(sample_sizes))
    mean_pool_errors_001=np.zeros(len(sample_sizes))
    mean_grulsif_errors_01=np.zeros(len(sample_sizes))
    mean_pool_errors_01=np.zeros(len(sample_sizes))
    mean_grulsif_errors_05=np.zeros(len(sample_sizes))
    mean_pool_errors_05=np.zeros(len(sample_sizes))
    ################# Convergence L2
    
    std_grulsif_errors_001=np.zeros(len(sample_sizes))
    std_pool_errors_001=np.zeros(len(sample_sizes))
    std_grulsif_errors_01=np.zeros(len(sample_sizes))
    std_pool_errors_01=np.zeros(len(sample_sizes))
    std_grulsif_errors_05=np.zeros(len(sample_sizes))
    std_pool_errors_05=np.zeros(len(sample_sizes))

    for i in range(len(sample_sizes)):
        
        alpha=0.01

        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_errors_001= pickle.load(input_file)
        
        
        if len(grulsif_errors_001)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in grulsif_errors_001])
            std_grulsif_errors_001[i]=np.std(mean_cost_function)
            mean_grulsif_errors_001[i]=np.mean(mean_cost_function)        
        else:
            mean_grulsif_errors_001[i]=np.mean(grulsif_errors_001)
        
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_errors_001= pickle.load(input_file)
        
        if len(pool_errors_001)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in pool_errors_001])
            std_pool_errors_001[i]=np.std(mean_cost_function)
            mean_pool_errors_001[i]=np.mean(mean_cost_function)        
        else:
            mean_pool_errors_001[i]=np.mean(pool_errors_001)
    
        alpha=0.1

        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_errors_01= pickle.load(input_file)
        
        if len(grulsif_errors_01)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in grulsif_errors_01])
            std_grulsif_errors_01[i]=np.std(mean_cost_function)
            mean_grulsif_errors_01[i]=np.mean(mean_cost_function)        
        else:
            mean_grulsif_errors_01[i]=np.mean(grulsif_errors_01)
    
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_errors_01= pickle.load(input_file)
        
        if len(pool_errors_01)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in pool_errors_01])
            std_pool_errors_01[i]=np.std(mean_cost_function)
            mean_pool_errors_01[i]=np.mean(mean_cost_function)        
        else:
            mean_pool_errors_01[i]=np.mean(pool_errors_01)
    
        alpha=0.5

        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_errors_05= pickle.load(input_file)
        
        if len(grulsif_errors_05)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in grulsif_errors_05])
            std_grulsif_errors_05[i]=np.std(mean_cost_function)
            mean_grulsif_errors_05[i]=np.mean(mean_cost_function)        
        else:
            mean_grulsif_errors_05[i]=np.mean(grulsif_errors_05)
    
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_errors_05= pickle.load(input_file)
        
        if len(pool_errors_05)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in pool_errors_05])
            std_pool_errors_05[i]=np.std(mean_cost_function)
            mean_pool_errors_05[i]=np.mean(mean_cost_function)        
        else:
            mean_pool_errors_05[i]=np.mean(pool_errors_05)
      
    fig = plt.figure()
    plt.plot(sample_sizes,np.log10(mean_grulsif_errors_001),label=r'GRULSIF $\alpha=0.01$', linestyle='solid')
    plt.fill_between(sample_sizes,np.log10(mean_grulsif_errors_001 - std_grulsif_errors_001), np.log10(mean_grulsif_errors_001 + std_grulsif_errors_001),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_pool_errors_001),label=r'Pool $\alpha=0.01$', linestyle='--')
    plt.fill_between(sample_sizes,np.log10(mean_pool_errors_001 - std_pool_errors_001), np.log10(mean_pool_errors_001 + std_pool_errors_001),alpha=0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"$n=n^{'}$",fontsize=20)
    plt.ylabel(r'$\log(P^{\alpha}[(\mathbf{\hat{f}}- \mathbf{r}^{\alpha})^2])$',fontsize=20)
    plt.tight_layout()

    plt.savefig(results_directory+"/"+experiment+f"convergence_L2_n_nodes_{n_nodes}_alpha_001.pdf")

    fig = plt.figure()
    plt.plot(sample_sizes,np.log10(mean_grulsif_errors_01),label=r'GRULSIF $\alpha=0.01$', linestyle='solid')
    plt.fill_between(sample_sizes,np.log10(mean_grulsif_errors_01 - std_grulsif_errors_01), np.log10(mean_grulsif_errors_01 + std_grulsif_errors_01),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_pool_errors_01),label=r'Pool $\alpha=0.01$', linestyle='--')
    plt.fill_between(sample_sizes,np.log10(mean_pool_errors_01 - std_pool_errors_01), np.log10(mean_pool_errors_01 + std_pool_errors_01),alpha=0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"$n=n^{'}$",fontsize=20)
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"convergence_L2_n_nodes_{n_nodes}_alpha_01.pdf")

    fig = plt.figure()
    plt.plot(sample_sizes,np.log10(mean_grulsif_errors_05),label=r'GRULSIF $\alpha=0.01$', linestyle='solid')
    plt.fill_between(sample_sizes,np.log10(mean_grulsif_errors_05 - std_grulsif_errors_05), np.log10(mean_grulsif_errors_05 + std_grulsif_errors_05),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_pool_errors_05),label=r'Pool $\alpha=0.01$', linestyle='--')
    plt.fill_between(sample_sizes,np.log10(mean_pool_errors_05 - std_pool_errors_05), np.log10(mean_pool_errors_05 + std_pool_errors_05),alpha=0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"$n=n^{'}$",fontsize=20)
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"convergence_L2_n_nodes_{n_nodes}_alpha_05.pdf")
    
    
def plot_convergence_boxplot_fdiv_alphas(results_directory,experiment,sample_sizes):
    ############ This function generates the box plots and heatmaps presented in the paper, 
    ############ showing the approximation to the real f-divergence values.

    ### Input
    # results_directory: the directory where the plots will be saved.
    # experiment: the experiment being studied: '1A', '1B', '2A', '2B', '2C'.
    # sample_sizes: sample sizes that were explored in the experiments.    
    
    try: 
        if experiment not in ["1A","1B","2A","2B","2C"]: 
            raise ValueError(F"Experiment should be 1A,1B,2A,2B or 2C")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)   
        
        
    alpha=0.1
    n_nodes=100
    
    if experiment=="1A":  
        
        real_PE_div_001=estimate_real_f_div(experiment,n_nodes,alpha=0.01)
        real_PE_div_001=[real_PE_div_001[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_001=np.median(np.vstack(real_PE_div_001),axis=1)
 
        real_PE_div_01=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
        real_PE_div_01=[real_PE_div_01[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_01=np.median(np.vstack(real_PE_div_01),axis=1)

        real_PE_div_05=estimate_real_f_div(experiment,n_nodes,alpha=0.5)
        real_PE_div_05=[real_PE_div_05[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_05=np.median(np.vstack(real_PE_div_05),axis=1)
        
    elif experiment=="1B": 
        
        real_PE_div_001=estimate_real_f_div(experiment,n_nodes,alpha=0.01)
        real_PE_div_001=[real_PE_div_001[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_001=np.median(np.vstack(real_PE_div_001),axis=1)

        real_PE_div_01=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
        real_PE_div_01=[real_PE_div_01[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_01=np.median(np.vstack(real_PE_div_01),axis=1)

        real_PE_div_05=estimate_real_f_div(experiment,n_nodes,alpha=0.5)
        real_PE_div_05=[real_PE_div_05[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_05=np.median(np.vstack(real_PE_div_05),axis=1)
         
       
    elif experiment=="2A": 
        
        G,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref=100,N_test=100)
        
        
        real_PE_div_001=estimate_real_f_div(experiment,n_nodes,alpha=0.01)
        real_PE_div_01=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
        real_PE_div_05=estimate_real_f_div(experiment,n_nodes,alpha=0.5)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))

        A=np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2)],[np.sin(-np.pi/2),np.cos(-np.pi/2)]])
        n_nodes=int(N_1*N_2)
        coordinates=np.zeros(shape=(n_nodes,2))
        for i in range(n_nodes):
            n_row=int(i/N_2)
            n_col=i % N_2
            coordinates[i,:]=A.dot(np.array([n_row,n_col])-np.array([N_1/2,N_2/2]))
        coordinates*=0.1
        
        G.set_coordinates(coordinates)  
        
        
    elif experiment=="2B":    
        
        d=2
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref=100,N_test=100,d=d)
        non_affected_nodes=set(np.arange(n_nodes))-set(affected_nodes)
        real_PE_div_001=estimate_real_f_div(experiment,n_nodes,alpha=0.01,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_001[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_001[list(non_affected_nodes)],axis=0) 
        real_PE_div_001=aux_real_PE_div
        
        real_PE_div_01=estimate_real_f_div(experiment,n_nodes,alpha=0.1,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_01[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_01[list(non_affected_nodes)],axis=0) 
        real_PE_div_01=aux_real_PE_div
        
        real_PE_div_05=estimate_real_f_div(experiment,n_nodes,alpha=0.5,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_05[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_05[list(non_affected_nodes)],axis=0) 
        real_PE_div_05=aux_real_PE_div

    elif experiment=="2C":    
        
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref=100,N_test=100,d=d)
        non_affected_nodes=set(np.arange(n_nodes))-set(affected_nodes)
        real_PE_div_001=estimate_real_f_div(experiment,n_nodes,alpha=0.01,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_001[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_001[list(non_affected_nodes)],axis=0) 
        real_PE_div_001=aux_real_PE_div
        
        real_PE_div_01=estimate_real_f_div(experiment,n_nodes,alpha=0.1,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_01[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_01[list(non_affected_nodes)],axis=0) 
        real_PE_div_01=aux_real_PE_div
        
        real_PE_div_05=estimate_real_f_div(experiment,n_nodes,alpha=0.5,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_05[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_05[list(non_affected_nodes)],axis=0) 
        real_PE_div_05=aux_real_PE_div
        
    PEARSON_cost_boxplot_grulsif_001=[] 
    PEARSON_cost_boxplot_pool_001=[]
    PEARSON_cost_boxplot_grulsif_01=[] 
    PEARSON_cost_boxplot_pool_01=[]
    PEARSON_cost_boxplot_grulsif_05=[] 
    PEARSON_cost_boxplot_pool_05=[]
    

    for i in range(len(sample_sizes)):
    
        N=sample_sizes[i]
        N_ref=N
        N_test=N
    
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test)
        
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test)
        
            
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_5_LRE(n_nodes,N_ref,N_test)
            
        elif experiment=="2B":
            d=2
            _,data_ref,data_test,affected_nodes=generate_experiment_6_LRE(n_nodes,N_ref,N_test,d=d)
            
        elif experiment=="2C":
            d=10
            _,data_ref,data_test,affected_nodes=generate_experiment_6_LRE(n_nodes,N_ref,N_test,d=d)
        
        alpha=0.01
    
        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_model_001= pickle.load(input_file)
        
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_model_001= pickle.load(input_file)
        
        alpha=0.1
    
        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_model_01= pickle.load(input_file)
        
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_model_01= pickle.load(input_file)
        
        alpha=0.5
    
        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_model_05= pickle.load(input_file)
        
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_model_05= pickle.load(input_file)
            
         
        f_div_grulsif_001=[]
        f_div_pool_001=[]
        f_div_grulsif_01=[]
        f_div_pool_01=[]
        f_div_grulsif_05=[]
        f_div_pool_05=[]
        
        for j in range(len( grulsif_model_05)):
   
            f_div_grulsif_001.append(f_div(grulsif_model_001[j]['theta'],grulsif_model_001[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.01,model="grulsif")) 
            f_div_pool_001.append(f_div(pool_model_001[j]['theta'],pool_model_001[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.01,model="pool"))
   
            f_div_grulsif_01.append(f_div(grulsif_model_01[j]['theta'],grulsif_model_01[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.1,model="grulsif")) 
            f_div_pool_01.append(f_div(pool_model_01[j]['theta'],pool_model_01[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.1,model="pool"))
    
            f_div_grulsif_05.append(f_div(grulsif_model_05[j]['theta'],grulsif_model_05[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.5,model="grulsif")) 
            f_div_pool_05.append(f_div(pool_model_05[j]['theta'],pool_model_05[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.5,model="pool"))
        
        
        mean_real_fdiv_001=np.mean(real_PE_div_001)
        mean_real_fdiv_01=np.mean(real_PE_div_01)
        mean_real_fdiv_05=np.mean(real_PE_div_05)
        
       
        distance_f_div_grulsif_001=[np.exp(-1.0*(np.mean(grulsif)-mean_real_fdiv_001)**2) for grulsif in f_div_grulsif_001]
        distance_f_div_pool_001=[np.exp(-1.0*(np.mean(pool)-mean_real_fdiv_001)**2) for pool in f_div_pool_001]
       
        distance_f_div_grulsif_01=[np.exp(-1.0*(np.mean(grulsif)-mean_real_fdiv_01)**2) for grulsif in f_div_grulsif_01]
        distance_f_div_pool_01=[np.exp(-1.0*(np.mean(pool)-mean_real_fdiv_01)**2) for pool in f_div_pool_01]
        
        distance_f_div_grulsif_05=[np.exp(-1.0*(np.mean(grulsif)-mean_real_fdiv_05)**2) for grulsif in f_div_grulsif_05]
        distance_f_div_pool_05=[np.exp(-1.0*(np.mean(pool)-mean_real_fdiv_05)**2) for pool in f_div_pool_05]
               
        f_div_grulsif_001=f_div_grulsif_001[np.argmax(distance_f_div_grulsif_001)]
        f_div_pool_001=f_div_pool_001[np.argmax(distance_f_div_pool_001)]
        
        f_div_grulsif_01=f_div_grulsif_01[np.argmax(distance_f_div_grulsif_01)]
        f_div_pool_01=f_div_pool_01[np.argmax(distance_f_div_pool_01)]
        
        f_div_grulsif_05=f_div_grulsif_05[np.argmax(distance_f_div_grulsif_05)]
        f_div_pool_05=f_div_pool_05[np.argmax(distance_f_div_pool_05)]
 
 
        if experiment in ["1A","1B"]:
            
            PEARSON_cost_boxplot_grulsif_001.append([ f_div_grulsif_001[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_pool_001.append([ f_div_pool_001[int(25*i):int(25*(i+1))] for i in range(4)]) 
            
            PEARSON_cost_boxplot_grulsif_01.append([ f_div_grulsif_01[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_pool_01.append([ f_div_pool_01[int(25*i):int(25*(i+1))] for i in range(4)]) 
            
            PEARSON_cost_boxplot_grulsif_05.append([ f_div_grulsif_05[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_pool_05.append([ f_div_pool_05[int(25*i):int(25*(i+1))] for i in range(4)]) 
        
        elif experiment in ["2B","2C"]:
   
            PEARSON_cost_boxplot_grulsif_001.append([ f_div_grulsif_001[affected_nodes],f_div_grulsif_001[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_pool_001.append([ f_div_pool_001[affected_nodes],f_div_pool_001[list(non_affected_nodes)]]) 
    
            PEARSON_cost_boxplot_grulsif_01.append([ f_div_grulsif_01[affected_nodes],f_div_grulsif_01[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_pool_01.append([ f_div_pool_01[affected_nodes],f_div_pool_01[list(non_affected_nodes)]]) 
    
            PEARSON_cost_boxplot_grulsif_05.append([ f_div_grulsif_05[affected_nodes],f_div_grulsif_05[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_pool_05.append([ f_div_pool_05[affected_nodes],f_div_pool_05[list(non_affected_nodes)]]) 


        elif experiment in ["2A"]:
            PEARSON_cost_boxplot_grulsif_001.append(f_div_grulsif_001)
            PEARSON_cost_boxplot_pool_001.append(f_div_pool_001)
            
            PEARSON_cost_boxplot_grulsif_01.append(f_div_grulsif_01)
            PEARSON_cost_boxplot_pool_01.append(f_div_pool_01)
            
            PEARSON_cost_boxplot_grulsif_05.append(f_div_grulsif_05)
            PEARSON_cost_boxplot_pool_05.append(f_div_pool_05)
            

    ###################### Experiment 1A  
    if experiment=="1A":          
    
    ############### N_ref=50

        labels=["C1","C2","C3","C4"]

        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-6.0,3.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-6.0,3.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].tick_params(axis='y', labelsize=20)


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.3,1.3])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.3,1.3])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.3,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)

        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.3,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[5].tick_params(axis='x', labelsize=20)
        axs[5].xaxis.tick_top()
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        ############### N_ref=100

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-6.0,3.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-6.0,3.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.3,1.3])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.3,1.3])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.3,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[1],
                                vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.3,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)
    
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")


        ############### N=250

        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-6.0,3.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-6.0,3.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.3,1.3])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.3,1.3])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.3,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.3,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        ############# N=500

        Nref=500
        
        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[3],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-6.0,3.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[3],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-6.0,3.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[3],
                                vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.3,1.3])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)
        
        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[3],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.3,1.3])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.3,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[3],
                            vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.3,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")
    
    ###################### Experiment 1B  
    elif experiment=="1B":                  
    ############### N_ref=50

        labels=["C1","C2","C3","C4"]
        Nref=50
        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[0],
                                vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-20.0,10.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-20.0,10.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.0,1.7])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.0,1.7])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.2,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.2,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        axs[5].tick_params(axis='x', labelsize=20)
        axs[5].xaxis.tick_top()
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        ############################## N_ref=100

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-20.0,10.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-20.0,10.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[1],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.0,1.7])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.0,1.7])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.2,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.2,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")


        ################### N=250

        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-20.0,10.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-20.0,10.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.0,1.7])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)
        
        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.0,1.7])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.2,0.3])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.2,0.3])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

    ############################

        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-20.0,10.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[0].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[0].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-20.0,10.0])
        C1=axs[1].axhline(y=real_PE_div_001[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div_001[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[1].axhline(y=real_PE_div_001[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[1].axhline(y=real_PE_div_001[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-1.0,1.7])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[2].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[2].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)
        
        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-1.0,1.7])
        C1=axs[3].axhline(y=real_PE_div_01[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed", label="C1")
        C2=axs[3].axhline(y=real_PE_div_01[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[3].axhline(y=real_PE_div_01[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[3].axhline(y=real_PE_div_01[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.3,0.2])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[4].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[4].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.3,0.2])
        C1=axs[5].axhline(y=real_PE_div_05[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",  label="C1")
        C2=axs[5].axhline(y=real_PE_div_05[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        C3=axs[5].axhline(y=real_PE_div_05[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed", label="C3")
        C4=axs[5].axhline(y=real_PE_div_05[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed", label="C4")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")
        
    elif experiment=="2A":         
        
        fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,5))
        im1=G.plot_signal(real_PE_div_001,limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(real_PE_div_001,limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(real_PE_div_01,limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(real_PE_div_01,limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(real_PE_div_05,limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)
        im6=G.plot_signal(real_PE_div_05,limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[5],colorbar=False)
        axs[5].set_title(" ")
        axs[5].set_axis_off()
        cbar6 = fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_real_alphas.pdf")
        
        Nref=50

        fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif_001[0],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool_001[0],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_grulsif_01[0],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_pool_01[0],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(PEARSON_cost_boxplot_grulsif_05[0],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        im6=G.plot_signal(PEARSON_cost_boxplot_pool_05[0],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[5],colorbar=False)
        axs[5].set_title(" ")
        axs[5].set_axis_off()
        cbar6 = fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=20) 
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        Nref=100

        fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif_001[1],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool_001[1],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_grulsif_01[1],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_pool_01[1],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(PEARSON_cost_boxplot_grulsif_05[1],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        im6=G.plot_signal(PEARSON_cost_boxplot_pool_05[1],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[5],colorbar=False)
        axs[5].set_title(" ")
        axs[5].set_axis_off()
        cbar6 = fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=20) 
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        Nref=250

        fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif_001[2],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool_001[2],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_grulsif_01[2],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_pool_01[2],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(PEARSON_cost_boxplot_grulsif_05[2],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        im6=G.plot_signal(PEARSON_cost_boxplot_pool_05[2],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[5],colorbar=False)
        axs[5].set_title(" ")
        axs[5].set_axis_off()
        cbar6 = fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=20) 
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        Nref=500

        fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif_001[3],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool_001[3],limits=[0.0,np.quantile(real_PE_div_001,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_grulsif_01[3],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_pool_01[3],limits=[0.0,np.quantile(real_PE_div_01,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(PEARSON_cost_boxplot_grulsif_05[3],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        im6=G.plot_signal(PEARSON_cost_boxplot_pool_05[3],limits=[0.0,np.quantile(real_PE_div_05,0.75)],ax=axs[5],colorbar=False)
        axs[5].set_title(" ")
        axs[5].set_axis_off()
        cbar6 = fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=20) 
        plt.tight_layout()
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")    

    
    elif experiment=="2B":   
        ############## 50 observations

        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-8.0,2.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-8.0,2.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)   


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-2.0,1.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-2.0,1.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)   
    
   
        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[5].tick_params(axis='x', labelsize=20)
        axs[5].xaxis.tick_top()
        axs[5].tick_params(axis='y', labelsize=20)   
 
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")



        ############## 100 observations

        labels=["C",r"$C^{\complement}$"]

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-8.0,2.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-8.0,2.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot2 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-2.0,1.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-2.0,1.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
        
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)
    
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")


        ############## 250 observations

        labels=["C",r"$C^{\complement}$"]

        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-8.0,2.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-8.0,2.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-2.0,1.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[2],
                                vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-2.0,1.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
        
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)   
      
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        ############## 500 observations

        labels=["C",r"$C^{\complement}$"]
        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-8.0,2.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[3],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-8.0,2.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot2 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-2.0,1.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-2.0,1.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
    
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)  
       
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")
   
    elif experiment=="2C":   
        ############## 50 observations

        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.0,30.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.0,30.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)   


        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-3.0,4.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-3.0,4.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)   
    
   
        bplot5 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)

        bplot6 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[5].tick_params(axis='x', labelsize=20)
        axs[5].xaxis.tick_top()
        axs[5].tick_params(axis='y', labelsize=20)  
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")



        ############## 100 observations

        labels=["C",r"$C^{\complement}$"]

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.0,30.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.0,30.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot2 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-3.0,4.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-3.0,4.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
        
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")


        ############## 250 observations

        labels=["C",r"$C^{\complement}$"]

        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.0,30.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.0,30.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-3.0,4.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[2],
                                vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-3.0,4.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
        
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)

        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)   

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")

        ############## 500 observations

        labels=["C",r"$C^{\complement}$"]
        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=6,figsize=(20,5))
        fig.tight_layout(pad=3.0)

        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif_001[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.0,30.0])
        C1=axs[0].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool_001[3],
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.0,30.0])
        axs[1].axhline(y=real_PE_div_001[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div_001[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)   
        

        bplot2 = axs[2].boxplot(PEARSON_cost_boxplot_grulsif_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[2].set_ylim([-3.0,4.0])
        C1=axs[2].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[2].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot3 = axs[3].boxplot(PEARSON_cost_boxplot_pool_01[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[3].set_ylim([-3.0,4.0])
        axs[3].axhline(y=real_PE_div_01[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_01[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)   
    
   
        bplot4 = axs[4].boxplot(PEARSON_cost_boxplot_grulsif_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,0.5])
        C1=axs[4].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[4].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        bplot5 = axs[5].boxplot(PEARSON_cost_boxplot_pool_05[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[5].set_ylim([-0.5,0.5])
        axs[5].axhline(y=real_PE_div_05[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[5].axhline(y=real_PE_div_05[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        #axs[5].tick_params(axis='x', labelsize=20)
        axs[5].set_xticks([])
        axs[5].tick_params(axis='y', labelsize=20)  
       
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}_alphas.pdf")
   
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment",type=str) #### The experiment to be run

    args=parser.parse_args()  
    results_directory=args.results_directory
    experiment=args.experiment 
  
    sample_sizes=np.array((50,100,250,500))
    
    plot_convergence_L2_LRE_alphas(results_directory,experiment,sample_sizes)
    plot_convergence_boxplot_fdiv_alphas(results_directory,experiment,sample_sizes)



























    