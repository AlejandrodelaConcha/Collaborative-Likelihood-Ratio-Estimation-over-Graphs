# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title: plot_results_errors
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-26              
# This version:     2025-02-26
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal is to generate the plots appearing in the paper, where different models are used.
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


def plot_convergence_L2_LRE(results_directory,experiment,sample_sizes):
    
    ############ This function generates the L2-convergence plots presented in the paper. 

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
    

    mean_grulsif_errors=np.zeros(len(sample_sizes))
    mean_pool_errors=np.zeros(len(sample_sizes))
    mean_rulsif_errors=np.zeros(len(sample_sizes))
    mean_ulsif_errors=np.zeros(len(sample_sizes))
    mean_kliep_errors=np.zeros(len(sample_sizes))
    
    std_grulsif_errors=np.zeros(len(sample_sizes))
    std_pool_errors=np.zeros(len(sample_sizes))
    std_rulsif_errors=np.zeros(len(sample_sizes))
    std_ulsif_errors=np.zeros(len(sample_sizes))
    std_kliep_errors=np.zeros(len(sample_sizes))

    for i in range(len(sample_sizes)):

        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_errors= pickle.load(input_file)
                        
        if len(grulsif_errors)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in grulsif_errors])
            std_grulsif_errors[i]=np.std(mean_cost_function)
            mean_grulsif_errors[i]=np.mean(mean_cost_function)        
        else:
            mean_grulsif_errors[i]=np.mean(grulsif_errors)
    
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_errors= pickle.load(input_file)
            
        if len(pool_errors)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in pool_errors])
            std_pool_errors[i]=np.std(mean_cost_function)
            mean_pool_errors[i]=np.mean(mean_cost_function)        
        else:
            mean_pool_errors[i]=np.mean(grulsif_errors)
    
        
        
    
        rulsif_file=results_directory+"/"+f"Rulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        rulsif_file=rulsif_file.replace(".","")
        rulsif_file=rulsif_file+".pickle"

        with open(rulsif_file, "rb") as input_file:
            rulsif_errors= pickle.load(input_file)
            
        if len(rulsif_errors)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in rulsif_errors])
            std_rulsif_errors[i]=np.std(mean_cost_function)
            mean_rulsif_errors[i]=np.mean(mean_cost_function)        
        else:
            mean_rulsif_errors[i]=np.mean(rulsif_errors)
    
    
        ulsif_file=results_directory+"/"+f"Ulsif_experiment_{experiment}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        ulsif_file=ulsif_file.replace(".","")
        ulsif_file=ulsif_file+".pickle"

        with open(ulsif_file, "rb") as input_file:
            ulsif_errors= pickle.load(input_file)
        
        mean_ulsif_errors[i]=np.mean(ulsif_errors)
        
        if len(ulsif_errors)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in ulsif_errors])
            std_ulsif_errors[i]=np.std(mean_cost_function)
            mean_ulsif_errors[i]=np.mean(mean_cost_function)        
        else:
            mean_ulsif_errors[i]=np.mean(ulsif_errors)
    
    
        kliep_file=results_directory+"/"+f"Kliep_experiment_{experiment}_n_nodes_{n_nodes}_N_{sample_sizes[i]}_errors"
        kliep_file=kliep_file.replace(".","")
        kliep_file=kliep_file+".pickle"

        with open(kliep_file, "rb") as input_file:
            kliep_errors= pickle.load(input_file)
        
        if len(kliep_errors)>1:
            mean_cost_function=np.array([np.mean(errors) for errors in kliep_errors])
            std_kliep_errors[i]=np.std(mean_cost_function)
            mean_kliep_errors[i]=np.mean(mean_cost_function)        
        else:
            mean_kliep_errors[i]=np.mean(kliep_errors)
        
    #### Ploting LRE
        
    fig = plt.figure()
    plt.plot(sample_sizes,np.log10(mean_grulsif_errors),label=r'GRULSIF $\alpha=0.1$', linestyle='solid')
    plt.fill_between(sample_sizes,np.log10(mean_grulsif_errors - std_grulsif_errors), np.log10(mean_grulsif_errors + std_grulsif_errors),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_pool_errors),label=r'Pool $\alpha=0.1$', linestyle='--')
    plt.fill_between(sample_sizes,np.log10(mean_pool_errors - std_pool_errors), np.log10(mean_pool_errors + std_pool_errors),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_rulsif_errors),label=r'Rulsif $\alpha=0.1$', linestyle='dotted')
    plt.fill_between(sample_sizes,np.log10(mean_rulsif_errors - std_rulsif_errors), np.log10(mean_rulsif_errors + std_rulsif_errors),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_ulsif_errors),label="ULSIF", linestyle=(0,(5,10)))
    plt.plot(sample_sizes,np.log10(mean_kliep_errors),label="KLIEP",linestyle="dashdot")  
    plt.xlabel(r"$n=n^{'}$",fontsize=20)
    plt.ylabel(r'$\log(P^{\alpha}[(\mathbf{\hat{f}}- \mathbf{r}^{\alpha})^2])$',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"convergence_L2_n_nodes_{n_nodes}.pdf")
    
    fig = plt.figure()
    plt.plot(sample_sizes,np.log10(mean_grulsif_errors),label=r'GRULSIF $\alpha=0.1$', linestyle='solid')
    plt.fill_between(sample_sizes,np.log10(mean_grulsif_errors - std_grulsif_errors), np.log10(mean_grulsif_errors + std_grulsif_errors),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_pool_errors),label=r'Pool $\alpha=0.1$', linestyle='--')
    plt.fill_between(sample_sizes,np.log10(mean_pool_errors - std_pool_errors), np.log10(mean_pool_errors + std_pool_errors),alpha=0.2)
    plt.plot(sample_sizes,np.log10(mean_rulsif_errors),label=r'Rulsif $\alpha=0.1$', linestyle='dotted')
    plt.fill_between(sample_sizes,np.log10(mean_rulsif_errors - std_rulsif_errors), np.log10(mean_rulsif_errors + std_rulsif_errors),alpha=0.2)

    plt.xlabel(r"$n=n^{'}$",fontsize=20)
    plt.ylabel(r'$\log(P^{\alpha}[(\mathbf{\hat{f}}- \mathbf{r}^{\alpha})^2])$',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"convergence_L2_zoom_{n_nodes}.pdf")
    
    
def plot_convergence_boxplot_fdiv(results_directory,experiment,sample_sizes):
    
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

        real_PE_div=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
        real_PE_div=[real_PE_div[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div=np.median(np.vstack(real_PE_div),axis=1)
        
        real_PE_div_0=estimate_real_f_div(experiment,n_nodes,alpha=0.0)
        real_PE_div_0=[real_PE_div_0[int(25*i):int(25*(i+1))] for i in range(4)]
        real_PE_div_0=np.median(np.vstack(real_PE_div_0),axis=1)

        real_KL_div=estimate_real_f_div(experiment,n_nodes,type="KL")
        real_KL_div=[real_KL_div[int(25*i):int(25*(i+1))] for i in range(4)]
        real_KL_div=np.median(np.vstack(real_KL_div),axis=1)
    
    elif experiment=="1B": 
        
       real_PE_div=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
       real_PE_div=[real_PE_div[int(25*i):int(25*(i+1))] for i in range(4)]
       real_PE_div=np.median(np.vstack(real_PE_div),axis=1)
       
       real_PE_div_0=estimate_real_f_div(experiment,n_nodes,alpha=0.0)
       real_PE_div_0=[real_PE_div_0[int(25*i):int(25*(i+1))] for i in range(4)]
       real_PE_div_0=np.median(np.vstack(real_PE_div_0),axis=1)

       real_KL_div=estimate_real_f_div(experiment,n_nodes,type="KL")
       real_KL_div=[real_KL_div[int(25*i):int(25*(i+1))] for i in range(4)]
       real_KL_div=np.median(np.vstack(real_KL_div),axis=1)
     
    elif experiment=="2A":  
        real_PE_div=estimate_real_f_div(experiment,n_nodes,alpha=0.1)
        real_PE_div_0=estimate_real_f_div(experiment,n_nodes,alpha=0.0)
        real_KL_div=estimate_real_f_div(experiment,n_nodes,alpha=0.0,type="KL")
        G,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref=100,N_test=100)
        
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
        real_PE_div=estimate_real_f_div(experiment,n_nodes,alpha=0.1,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div[list(non_affected_nodes)],axis=0) 
        real_PE_div=aux_real_PE_div
        print(real_PE_div)

        real_PE_div_0=estimate_real_f_div(experiment,n_nodes,alpha=0.0,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_0[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_0[list(non_affected_nodes)],axis=0) 
        real_PE_div_0=aux_real_PE_div
        print(real_PE_div_0)

        real_KL_div=estimate_real_f_div(experiment,n_nodes,alpha=0.0,type="KL",d=d)
        aux_real_KL_div=np.zeros(2)
        aux_real_KL_div[0]=np.median(real_KL_div[affected_nodes],axis=0) 
        aux_real_KL_div[1]=np.median(real_KL_div[list(non_affected_nodes)],axis=0) 
        real_KL_div=aux_real_KL_div
        print(real_KL_div)
        
    elif experiment=="2C":    
        
        d=10
        _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref=100,N_test=100,d=d)
        non_affected_nodes=set(np.arange(n_nodes))-set(affected_nodes)
        real_PE_div=estimate_real_f_div(experiment,n_nodes,alpha=0.1,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div[list(non_affected_nodes)],axis=0) 
        real_PE_div=aux_real_PE_div
        print(real_PE_div)

        real_PE_div_0=estimate_real_f_div(experiment,n_nodes,alpha=0.0,d=d)
        aux_real_PE_div=np.zeros(2)
        aux_real_PE_div[0]=np.median(real_PE_div_0[affected_nodes],axis=0) 
        aux_real_PE_div[1]=np.median(real_PE_div_0[list(non_affected_nodes)],axis=0) 
        real_PE_div_0=aux_real_PE_div
        print(real_PE_div_0)

        real_KL_div=estimate_real_f_div(experiment,n_nodes,alpha=0.0,type="KL",d=d)
        aux_real_KL_div=np.zeros(2)
        aux_real_KL_div[0]=np.median(real_KL_div[affected_nodes],axis=0) 
        aux_real_KL_div[1]=np.median(real_KL_div[list(non_affected_nodes)],axis=0) 
        real_KL_div=aux_real_KL_div
        print(real_KL_div)
                
    PEARSON_cost_boxplot_grulsif=[] 
    PEARSON_cost_boxplot_pool=[]
    PEARSON_cost_boxplot_rulsif=[]
    PEARSON_cost_boxplot_ulsif=[]
    KL_cost_boxplot_kliep=[]

    for i in range(len(sample_sizes)):
        N=sample_sizes[i]
        N_ref=N
        N_test=N
        if experiment=="1A":
            _,data_ref,data_test,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test)
            
        elif experiment=="1B":
            _,data_ref,data_test,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test)
            
        elif experiment=="2A":
            _,data_ref,data_test,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
 
        elif experiment=="2B":
            d=2
            _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
            
        elif experiment=="2C":
            d=10
            _,data_ref,data_test,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
             
        grulsif_file=results_directory+"/"+f"Grulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        grulsif_file=grulsif_file.replace(".","")
        grulsif_file=grulsif_file+".pickle"

        with open(grulsif_file, "rb") as input_file:
            grulsif_model= pickle.load(input_file)
            
        pool_file=results_directory+"/"+f"Pool_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        pool_file=pool_file.replace(".","")
        pool_file=pool_file+".pickle"

        with open(pool_file, "rb") as input_file:
            pool_model= pickle.load(input_file)
            
        rulsif_file=results_directory+"/"+f"Rulsif_experiment_{experiment}_alpha_{alpha}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        rulsif_file=rulsif_file.replace(".","")
        rulsif_file=rulsif_file+".pickle"

        with open(rulsif_file, "rb") as input_file:
            rulsif_model= pickle.load(input_file)
            
        ulsif_file=results_directory+"/"+f"Ulsif_experiment_{experiment}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        ulsif_file=ulsif_file.replace(".","")
        ulsif_file=ulsif_file+".pickle"

        with open(ulsif_file, "rb") as input_file:
            ulsif_model= pickle.load(input_file)
            
      
        kliep_file=results_directory+"/"+f"Kliep_experiment_{experiment}_n_nodes_{n_nodes}_N_{N}_LRE_results"
        kliep_file=kliep_file.replace(".","")
        kliep_file=kliep_file+".pickle"

        with open(kliep_file, "rb") as input_file:
            kliep_model= pickle.load(input_file)
            
        f_div_grulsif=[]
        f_div_pool=[]
        f_div_rulsif=[]
        f_div_ulsif=[]
        f_div_kliep=[]
            
        for j in range(len( grulsif_model)):
            f_div_grulsif.append(f_div(grulsif_model[j]['theta'],grulsif_model[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.1,model="grulsif")) 
            f_div_pool.append(f_div(pool_model[j]['theta'],pool_model[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.1,model="pool"))
            f_div_rulsif.append(f_div(rulsif_model[j]['theta'],rulsif_model[j]['LREmodel_']['kernel'],data_ref,data_test,alpha=0.1,model="rulsif"))
            f_div_ulsif.append(f_div(ulsif_model[j]['theta'],ulsif_model[j]['LREmodel_']['kernel'],data_ref,data_test,model="ulsif"))
            f_div_kliep.append(f_div(kliep_model[j]['theta'],kliep_model[j]['LREmodel_']['kernel'],data_ref,data_test,model="kliep"))
        

        mean_real_PEdiv=np.mean(real_PE_div)
        mean_real_PEdiv_0=np.mean(real_PE_div_0)
        mean_real_KLdiv=np.mean(real_KL_div)

        distance_f_div_grulsif=[np.exp(-1.0*(np.mean(grulsif)-mean_real_PEdiv)**2) for grulsif in f_div_grulsif]
        distance_f_div_pool=[np.exp(-1.0*(np.mean(pool)-mean_real_PEdiv)**2) for pool in f_div_pool]
        distance_f_div_rulsif=[np.exp(-1.0*(np.mean(rulsif)-mean_real_PEdiv)**2) for rulsif in f_div_rulsif]
        distance_f_div_ulsif=[np.exp(-1.0*(np.mean(ulsif)-mean_real_PEdiv_0)**2) for ulsif in f_div_ulsif]
        distance_f_div_kliep=[np.exp(-1.0*(np.mean(kliep)-mean_real_KLdiv)**2) for kliep in f_div_kliep]        
       
        f_div_grulsif=f_div_grulsif[np.argmax(distance_f_div_grulsif)]
        f_div_pool=f_div_pool[np.argmax(distance_f_div_pool)]
        f_div_rulsif=f_div_rulsif[np.argmax(distance_f_div_rulsif)]
        f_div_ulsif=f_div_ulsif[np.argmax(distance_f_div_ulsif)] 
        f_div_kliep=f_div_kliep[np.argmax(distance_f_div_kliep)] 
            

        if experiment in ["1A","1B"]:
            PEARSON_cost_boxplot_grulsif.append([ f_div_grulsif[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_pool.append([ f_div_pool[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_rulsif.append([ f_div_rulsif[int(25*i):int(25*(i+1))] for i in range(4)]) 
            PEARSON_cost_boxplot_ulsif.append([ f_div_ulsif[int(25*i):int(25*(i+1))] for i in range(4)]) 
            KL_cost_boxplot_kliep.append([ f_div_kliep[int(25*i):int(25*(i+1))] for i in range(4)]) 

        elif experiment in ["2B","2C"]:
            PEARSON_cost_boxplot_grulsif.append([ f_div_grulsif[affected_nodes],f_div_grulsif[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_pool.append([ f_div_pool[affected_nodes],f_div_pool[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_rulsif.append([ f_div_rulsif[affected_nodes],f_div_rulsif[list(non_affected_nodes)]]) 
            PEARSON_cost_boxplot_ulsif.append([ f_div_ulsif[affected_nodes],f_div_ulsif[list(non_affected_nodes)]]) 
            KL_cost_boxplot_kliep.append([ f_div_kliep[affected_nodes],f_div_kliep[list(non_affected_nodes)]]) 
            
        elif experiment in ["2A"]:
            PEARSON_cost_boxplot_grulsif.append(f_div_grulsif)
            PEARSON_cost_boxplot_pool.append(f_div_pool)
            PEARSON_cost_boxplot_rulsif.append(f_div_rulsif)
            PEARSON_cost_boxplot_ulsif.append(f_div_ulsif)
            KL_cost_boxplot_kliep.append(f_div_kliep)
            
                 
    ###################### Experiment 1A
    
    
    if experiment=="1A":   
    ############# 50 observations
        labels=["C1","C2","C3","C4"]

        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.8,1.6])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.8,1.6])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.8,1.6])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)
        
        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.0,2.1])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[0],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.6])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")  
             
        ################## n 100

        labels=["C1","C2","C3","C4"]

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.8,1.6])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.8,1.6])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.8,1.6])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
#axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.0,2.30])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[1],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.6])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
#axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        ################ n=250
        labels=["C1","C2","C3","C4"]

        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.8,1.3])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.8,1.3])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
        
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.8,1.3])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.0,1.90])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed",label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot4= axs[4].boxplot(KL_cost_boxplot_kliep[2],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.3])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
#axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        ########### n=500

        labels=["C1","C2","C3","C4"]

        Nref=500
        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.8,1.3])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.8,1.3])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.8,1.3])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
#axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.0,1.90])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
#axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[3],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.3])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
#axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
        
        
    elif experiment=="1B":        
    ############# 50 observations
        labels=["C1","C2","C3","C4"]
        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[0],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.5,1.7])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[0],
                    vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.5,1.7])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
     
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[0],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.5,1.7])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
        axs[2].tick_params(axis='x', labelsize=15)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=15)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[0],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,4.00])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[0],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.7])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
      
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

    ################## n 100


        labels=["C1","C2","C3","C4"]

        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[1],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.5,1.7])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
    #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[1],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.5,1.7])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
     
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[1],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.5,1.7])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
    #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[1],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,4.00])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
    #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[1],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.7])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
      
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

    ################################### n=250

        labels=["C1","C2","C3","C4"]

        Nref=250
        
        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[2],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.5,1.7])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
    #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)

        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[2],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.5,1.7])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
    #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
     
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[2],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
    #axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.5,1.7])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
    #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[2],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,4.00])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red', linestyle="dashed",label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
    #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[2],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.7])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
    #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

    ############################### n=500

        labels=["C1","C2","C3","C4"]

        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[3],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-0.5,1.7])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[0].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[0].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[0].tick_params(axis='x', labelsize=20)
        axs[0].set_xticks([])
        axs[0].tick_params(axis='y', labelsize=20)
        
        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[3],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        #axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-0.5,1.7])
        C1=axs[1].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        C2=axs[1].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        C3=axs[1].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        C4=axs[1].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[1].tick_params(axis='x', labelsize=20)
        axs[1].set_xticks([])
        axs[1].tick_params(axis='y', labelsize=20)
     
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[3],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        #axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-0.5,1.7])
        axs[2].axhline(y=real_PE_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[2].axhline(y=real_PE_div[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[2].axhline(y=real_PE_div[3],xmin=0.75,xmax=1.00,color='green',linestyle="dashed",label="C4")
        #axs[2].tick_params(axis='x', labelsize=20)
        axs[2].set_xticks([])
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[3],
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,4.0])
        axs[3].axhline(y=real_PE_div_0[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div_0[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed", label="C2")
        axs[3].axhline(y=real_PE_div_0[2],xmin=0.5,xmax=0.75, color='red',linestyle="dashed",label="C3")
        axs[3].axhline(y=real_PE_div_0[3],xmin=0.75,xmax=1.00, color='green',linestyle="dashed",label="C4")
        #axs[3].tick_params(axis='x', labelsize=20)
        axs[3].set_xticks([])
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[3],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.8,1.7])
        axs[4].axhline(y=real_KL_div[0],xmin=0.00,xmax=0.25, color='red',linestyle="dashed",label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.25,xmax=0.5, color='green',linestyle="dashed",label="C2")
        axs[4].axhline(y=real_KL_div[2],xmin=0.5,xmax=0.75,  color='red',linestyle="dashed",label="C3")
        axs[4].axhline(y=real_KL_div[3],xmin=0.75,xmax=1.00,  color='green',linestyle="dashed",label="C4")
        #axs[4].tick_params(axis='x', labelsize=20)
        axs[4].set_xticks([])
        axs[4].tick_params(axis='y', labelsize=20)
      
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
        
    elif experiment=="2B":   
            
            ############# 50 observations   
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.5,1.2])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.5,1.2])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-1.5,1.2])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,1.2])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[0],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-1.5,1.2])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
    
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.5,1.2])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.5,1.2])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-1.5,1.2])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,1.2])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[1],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-1.5,1.2])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
            
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.5,1.2])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.5,1.2])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-1.5,1.2])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,1.2])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[2],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-1.5,1.2])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-1.5,1.2])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-1.5,1.2])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-1.5,1.2])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-1.5,1.2])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[3],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-1.5,1.2])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)

        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
        
    elif experiment=="2C":   
            
            ############# 50 observations   
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=50

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-3.0,4.0])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-3.0,4.0])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-3.0,4.0])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[0],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-0.5,4.0])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[0],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,5.5])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
    
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=100

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-3.0,4.0])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-3.0,4.0])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-3.0,4.0])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[1],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-0.5,4.0])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[1],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,5.0])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
            
        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=250

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-3.0,4.0])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-3.0,4.0])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-3.0,4.0])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[2],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-0.5,4.0])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[2],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,5.0])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
  
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        labels=[r"$C$",r"$C^{\complement}$"]
        Nref=500

        fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(20,5))
        fig.tight_layout(pad=3.0)
        bplot1 = axs[0].boxplot(PEARSON_cost_boxplot_grulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[0].set_ylim([-3.0,4.0])
        C1=axs[0].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        C2=axs[0].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(axis='y', labelsize=20)


        bplot2 = axs[1].boxplot(PEARSON_cost_boxplot_pool[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[1].set_title(r'POOL $\alpha=0.1$', fontsize=75)
        axs[1].set_ylim([-3.0,4.0])
        axs[1].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[1].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].xaxis.tick_top()
        axs[1].tick_params(axis='y', labelsize=20)
 
        bplot3 = axs[2].boxplot(PEARSON_cost_boxplot_rulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
#axs[2].set_title(r'RULSIF $\alpha=0.1$', fontsize=75)
        axs[2].set_ylim([-3.0,4.0])
        axs[2].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[2].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].xaxis.tick_top()
        axs[2].tick_params(axis='y', labelsize=20)

        bplot4 =axs[3].boxplot(PEARSON_cost_boxplot_ulsif[3],
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[3].set_ylim([-0.5,4.0])
        axs[3].axhline(y=real_PE_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[3].axhline(y=real_PE_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[3].tick_params(axis='x', labelsize=20)
        axs[3].xaxis.tick_top()
        axs[3].tick_params(axis='y', labelsize=20)

        bplot5= axs[4].boxplot(KL_cost_boxplot_kliep[3],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
        axs[4].set_ylim([-0.5,5.0])
        axs[4].axhline(y=real_KL_div[0],xmin=0.10,xmax=0.40, color='red',linestyle="dashed", label="C1")
        axs[4].axhline(y=real_KL_div[1],xmin=0.6,xmax=0.90, color='green',linestyle="dashed",label="C2")
        axs[4].tick_params(axis='x', labelsize=20)
        axs[4].xaxis.tick_top()
        axs[4].tick_params(axis='y', labelsize=20)
       
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")
        
    elif experiment=="2A":         
        
        fig,axs = plt.subplots(nrows=1,ncols=5,figsize=(30,5))
        im1=G.plot_signal(real_PE_div,limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(real_PE_div,limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(real_PE_div,limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(real_PE_div_0,limits=[0.0,np.quantile(real_PE_div_0,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(real_KL_div,limits=[0.0,np.quantile(real_KL_div,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_real.pdf")
        
        Nref=50

        fig,axs = plt.subplots(nrows=1,ncols=5,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif[0],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool[0],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_rulsif[0],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_ulsif[0],limits=[0.0,np.quantile(real_PE_div_0,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(KL_cost_boxplot_kliep[0],limits=[0.0,np.quantile(real_KL_div,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        Nref=100

        fig,axs = plt.subplots(nrows=1,ncols=5,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif[1],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool[1],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_rulsif[1],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_ulsif[1],limits=[0.0,np.quantile(real_PE_div_0,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(KL_cost_boxplot_kliep[1],limits=[0.0,np.quantile(real_KL_div,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        Nref=250

        fig,axs = plt.subplots(nrows=1,ncols=5,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif[2],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool[2],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_rulsif[2],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_ulsif[2],limits=[0.0,np.quantile(real_PE_div_0,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(KL_cost_boxplot_kliep[2],limits=[0.0,np.quantile(real_KL_div,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
        plt.tight_layout()
        
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")

        Nref=500

        fig,axs = plt.subplots(nrows=1,ncols=5,figsize=(30,5))
        im1=G.plot_signal(PEARSON_cost_boxplot_grulsif[3],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[0],colorbar=False)
        axs[0].set_title(" ")
        axs[0].set_axis_off()
        cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=20)
        im2=G.plot_signal(PEARSON_cost_boxplot_pool[3],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[1],colorbar=False)
        axs[1].set_title(" ")
        axs[1].set_axis_off()
        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=20)
        im3=G.plot_signal(PEARSON_cost_boxplot_rulsif[3],limits=[0.0,np.quantile(real_PE_div,0.75)],ax=axs[2],colorbar=False)
        axs[2].set_title(" ")
        axs[2].set_axis_off()
        cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=20)  
        im4=G.plot_signal(PEARSON_cost_boxplot_ulsif[3],limits=[0.0,np.quantile(real_PE_div_0,0.75)],ax=axs[3],colorbar=False)
        axs[3].set_title(" ")
        axs[3].set_axis_off()
        cbar4 = fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=20)
        im5=G.plot_signal(KL_cost_boxplot_kliep[3],limits=[0.0,np.quantile(real_KL_div,0.75)],ax=axs[4],colorbar=False)
        axs[4].set_title(" ")
        axs[4].set_axis_off()
        cbar5 = fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=20)  
      
        plt.tight_layout()
        plt.savefig(results_directory+"/"+experiment+f"_fdiv_boxplot_n_nodes_{n_nodes}_Nref_{Nref}.pdf")




if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment",type=str) #### The experiment to be run

    args=parser.parse_args()  
    results_directory=args.results_directory
    experiment=args.experiment 
    
    sample_sizes=np.array((50,100,250,500))
    
    plot_convergence_L2_LRE(results_directory,experiment,sample_sizes)
    plot_convergence_boxplot_fdiv(results_directory,experiment,sample_sizes)









