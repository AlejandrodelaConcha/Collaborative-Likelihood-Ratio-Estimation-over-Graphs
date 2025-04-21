# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  plot_AFROC_alphas.py 
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2024-03-01              
# This version:     2024-03-01
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to replicate the comparision of ctst and pool described in the paper. 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: pickle, argparse, Experiments.experiments_two_sample_test, pandas
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: AFROC, ROC, AUC, TPR, FPR, FWER
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from Experiments.experiments_two_sample_test import generate_experiment_1,generate_experiment_2,generate_experiment_3,generate_experiment_4
import pandas as pd
import copy

def plot_AFROC_ROC_synthetic_experiments_alphas(results_directory,experiment,N):
################ This function produces the AFROC and ROC curves associated with the Synthetic Experiments.

## Input:
# results_directory: directory where the results will be stored.
# experiment: the experiment to be run.
# N: the number of observations per node.

## Output:
# The code generates 4 files: two files with the plots of the AFROC and ROC curves comparing the different methods, 
# and two CSV files containing the AUC of the different methods.

       
    N_ref=N
    alpha=0.01
    
    AFROC_c2st_file_001=results_directory+"/"+f"AFROC_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_c2st_file_001=AFROC_c2st_file_001.replace(".","")
    AFROC_c2st_file_001=AFROC_c2st_file_001+".pickle"

    with open(AFROC_c2st_file_001, "rb") as input_file:
        AFROC_c2st_001= pickle.load(input_file)
     
    AFROC_pool_file_001=results_directory+"/"+f"AFROC_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_pool_file_001=AFROC_pool_file_001.replace(".","")
    AFROC_pool_file_001=AFROC_pool_file_001+".pickle"

    with open(AFROC_pool_file_001, "rb") as input_file:
        AFROC_pool_001= pickle.load(input_file)
        
    N_ref=N
    alpha=0.1
    
    AFROC_c2st_file_01=results_directory+"/"+f"AFROC_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_c2st_file_01=AFROC_c2st_file_01.replace(".","")
    AFROC_c2st_file_01=AFROC_c2st_file_01+".pickle"

    with open(AFROC_c2st_file_01, "rb") as input_file:
        AFROC_c2st_01= pickle.load(input_file)
     
    AFROC_pool_file_01=results_directory+"/"+f"AFROC_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_pool_file_01=AFROC_pool_file_01.replace(".","")
    AFROC_pool_file_01=AFROC_pool_file_01+".pickle"

    with open(AFROC_pool_file_01, "rb") as input_file:
        AFROC_pool_01= pickle.load(input_file)
        
    N_ref=N
    alpha=0.5
    
    AFROC_c2st_file_05=results_directory+"/"+f"AFROC_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_c2st_file_05=AFROC_c2st_file_05.replace(".","")
    AFROC_c2st_file_05=AFROC_c2st_file_05+".pickle"

    with open(AFROC_c2st_file_05, "rb") as input_file:
        AFROC_c2st_05= pickle.load(input_file)
     
    AFROC_pool_file_05=results_directory+"/"+f"AFROC_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_pool_file_05=AFROC_pool_file_05.replace(".","")
    AFROC_pool_file_05=AFROC_pool_file_05+".pickle"

    with open(AFROC_pool_file_05, "rb") as input_file:
        AFROC_pool_05= pickle.load(input_file)
        
    index_c2st_001=np.where(AFROC_c2st_001["FWR"]<=0.05)[0]
    index_pool_001=np.where(AFROC_pool_001["FWR"]<=0.05)[0]
    index_c2st_01=np.where(AFROC_c2st_01["FWR"]<=0.05)[0]
    index_pool_01=np.where(AFROC_pool_01["FWR"]<=0.05)[0]
    index_c2st_05=np.where(AFROC_c2st_05["FWR"]<=0.05)[0]
    index_pool_05=np.where(AFROC_pool_05["FWR"]<=0.05)[0]
   
    AFROC_c2st_FWR_001=np.hstack((0.05,AFROC_c2st_001["FWR"][index_c2st_001]))
    AFROC_c2st_TPR_001=np.hstack((np.max(AFROC_c2st_001["TPR"][index_c2st_001]),AFROC_c2st_001["TPR"][index_c2st_001]))
    AFROC_pool_FWR_001=np.hstack((0.05,AFROC_pool_001["FWR"][index_pool_001]))
    AFROC_pool_TPR_001=np.hstack((np.max(AFROC_pool_001["TPR"][index_pool_001]),AFROC_pool_001["TPR"][index_pool_001]))
    AFROC_c2st_FWR_01=np.hstack((0.05,AFROC_c2st_01["FWR"][index_c2st_01]))
    AFROC_c2st_TPR_01=np.hstack((np.max(AFROC_c2st_01["TPR"][index_c2st_01]),AFROC_c2st_01["TPR"][index_c2st_01]))
    AFROC_pool_FWR_01=np.hstack((0.05,AFROC_pool_01["FWR"][index_pool_01]))
    AFROC_pool_TPR_01=np.hstack((np.max(AFROC_pool_01["TPR"][index_pool_01]),AFROC_pool_01["TPR"][index_pool_01]))
    AFROC_c2st_FWR_05=np.hstack((0.05,AFROC_c2st_05["FWR"][index_c2st_05]))
    AFROC_c2st_TPR_05=np.hstack((np.max(AFROC_c2st_05["TPR"][index_c2st_05]),AFROC_c2st_05["TPR"][index_c2st_05]))
    AFROC_pool_FWR_05=np.hstack((0.05,AFROC_pool_05["FWR"][index_pool_05]))
    AFROC_pool_TPR_05=np.hstack((np.max(AFROC_pool_05["TPR"][index_pool_05]),AFROC_pool_05["TPR"][index_pool_05]))
  
    fig = plt.figure(figsize=(12,10))
    plt.plot(AFROC_c2st_FWR_001,AFROC_c2st_TPR_001,label=r'C2ST $\alpha=0.01$',linestyle='solid',linewidth=5.0)
    plt.plot(AFROC_pool_FWR_001,AFROC_pool_TPR_001,label=r'POOL $\alpha=0.01$', linestyle='--',linewidth=5.0)
    plt.plot(AFROC_c2st_FWR_01,AFROC_c2st_TPR_01,label=r'C2ST $\alpha=0.1$', linestyle='dotted',linewidth=5.0)
    plt.plot(AFROC_pool_FWR_01,AFROC_pool_TPR_01,label=r'POOL $\alpha=0.1$',linestyle=(0,(5,10)),linewidth=5.0)
    plt.plot(AFROC_c2st_FWR_05,AFROC_c2st_TPR_05,label=r'C2ST $\alpha=0.5$',linestyle="dashdot",linewidth=5.0)
    plt.plot(AFROC_pool_FWR_05,AFROC_pool_TPR_05,label=r'POOL $\alpha=0.5$',linestyle=(0,(3,5,1,5)),linewidth=5.0)      
    plt.ylim(-0.1, 1.1)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=100)
    plt.xlabel("FWER",fontsize=50)
    plt.ylabel("TPR",fontsize=50)
    plt.legend(loc='lower right') 
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"AFROC_N_{N_ref}_FW_FPR_alphas.pdf")
     
    AUC_FWR_c2st_001=metrics.auc(AFROC_c2st_FWR_001,AFROC_c2st_TPR_001)/0.05
    AUC_FWR_pool_001=metrics.auc(AFROC_pool_FWR_001,AFROC_pool_TPR_001)/0.05
    AUC_FWR_c2st_01=metrics.auc(AFROC_c2st_FWR_01,AFROC_c2st_TPR_01)/0.05
    AUC_FWR_pool_01=metrics.auc(AFROC_pool_FWR_01,AFROC_pool_TPR_01)/0.05
    AUC_FWR_c2st_05=metrics.auc(AFROC_c2st_FWR_05,AFROC_c2st_TPR_05)/0.05
    AUC_FWR_pool_05=metrics.auc(AFROC_pool_FWR_05,AFROC_pool_TPR_05)/0.05
    

    AUC_FWR=np.vstack((AUC_FWR_c2st_001,AUC_FWR_pool_001,AUC_FWR_c2st_01,AUC_FWR_pool_01,AUC_FWR_c2st_05,AUC_FWR_pool_05))
    AUC_FWR=pd.DataFrame(AUC_FWR,index=["C2ST_001","POOL_001","C2ST_01","POOL_01","C2ST_05","POOL_05"])
    AUC_FWR.to_csv(results_directory+"/"+experiment+f"AUC_FWR_N_{N_ref}_alphas.csv") 
         
    FPR_c2st_001=AFROC_c2st_001["FPR"]
    TPR_c2st_001=AFROC_c2st_001["TPR"]
    FPR_pool_001=AFROC_pool_001["FPR"]
    TPR_pool_001=AFROC_pool_001["TPR"]
    FPR_c2st_01=AFROC_c2st_01["FPR"]
    TPR_c2st_01=AFROC_c2st_01["TPR"]
    FPR_pool_01=AFROC_pool_01["FPR"]
    TPR_pool_01=AFROC_pool_01["TPR"]
    FPR_c2st_05=AFROC_c2st_05["FPR"]
    TPR_c2st_05=AFROC_c2st_05["TPR"]
    FPR_pool_05=AFROC_pool_05["FPR"]
    TPR_pool_05=AFROC_pool_05["TPR"]

    fig = plt.figure(figsize=(12,10))
    plt.plot(FPR_c2st_001,TPR_c2st_001,label=r'C2ST $\alpha=0.01$',linestyle='solid')
    plt.plot(FPR_pool_001,TPR_pool_001,label=r'POOL $\alpha=0.01$', linestyle='--')
    plt.plot(FPR_c2st_01,TPR_c2st_01,label=r'C2ST $\alpha=0.1$', linestyle='dotted')
    plt.plot(FPR_pool_01,TPR_pool_01,label=r'POOL $\alpha=0.1$', linestyle=(0,(5,10)))
    plt.plot(FPR_c2st_05,TPR_c2st_05,label=r'C2ST $\alpha=0.5$',linestyle="dashdot")
    plt.plot(FPR_pool_05,TPR_pool_05,label=r'POOL $\alpha=0.5$',linestyle=(0,(3,5,1,5)))      
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("FPR",fontsize=50)
    plt.ylabel("TPR",fontsize=50)
    plt.savefig(results_directory+"/"+experiment+f"ROC_N_{N_ref}_alphas.pdf")

    AUC_c2st_001=metrics.auc(FPR_c2st_001,TPR_c2st_001)
    AUC_pool_001=metrics.auc(FPR_pool_001,TPR_pool_001)
    AUC_c2st_01=metrics.auc(FPR_c2st_01,TPR_c2st_01)
    AUC_pool_01=metrics.auc(FPR_pool_01,TPR_pool_01)
    AUC_c2st_05=metrics.auc(FPR_c2st_05,TPR_c2st_05)
    AUC_pool_05=metrics.auc(FPR_pool_05,TPR_pool_05)
    
    AUC=np.vstack((AUC_c2st_001,AUC_pool_001,AUC_c2st_01,AUC_pool_01,AUC_c2st_05,AUC_pool_05))
    AUC=pd.DataFrame(AUC,index=["C2ST_001","POOL_001","C2ST_01","POOL_01","C2ST_05","POOL_05"])
    AUC.to_csv(results_directory+"/"+experiment+f"AUC_N_{N_ref}_alphas.csv")   
    
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory",type=str) #### Dictionary where the results will be saved
    parser.add_argument("--experiment",type=str) #### The name of the model to be run
    parser.add_argument("--N",type=int) #### The experiment to be run
    
    args=parser.parse_args()
    
    results_directory=args.results_directory
    experiment=args.experiment
    N=args.N

    plot_AFROC_ROC_synthetic_experiments_alphas(results_directory,experiment,N)

