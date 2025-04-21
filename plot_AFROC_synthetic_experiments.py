# ----------------------------------------------------------------------------------------------------------------------
# Title:           plot_AFROC_synthetic_experiments.py 
# Author(s):       Alejandro de la Concha
# Initial version: 2024-01-15
# Last modified:   2025-02-28              
# This version:    2025-02-28
# ----------------------------------------------------------------------------------------------------------------------
# Objective(s): 
#   - Plot the AFROC curve for the hypothesis testing problems compared in the paper.
# ----------------------------------------------------------------------------------------------------------------------
# Library dependencies: 
#   - pickle
#   - argparse
#   - pandas
#   - Experiments.experiments_two_sample_test (custom module)
# ----------------------------------------------------------------------------------------------------------------------
# Keywords: AFROC, ROC, AUC, TPR, FPR, FWER
# ----------------------------------------------------------------------------------------------------------------------

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from Experiments.experiments_two_sample_test import generate_experiment_1,generate_experiment_2,generate_experiment_3,generate_experiment_4
import pandas as pd
import copy


def plot_AFROC_ROC_synthetic_experiments(results_directory,experiment,N):
################ This function produce the AFROC and ROC curve associated to the Synthetic Experiments
## Input
# results_directory: directory where the results will be stored
# experiment: the experiment to be run 
# N: the number of observations per node
## Output  
## Output
# The code generates 4 files: two files with the plots of the AFROC and ROC curves comparing the different methods
# 2 more, csv where the AUC of the different methods.
   
       
    N_ref=N
    alpha=0.1
    
    AFROC_c2st_file=results_directory+"/"+f"AFROC_c2st_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_c2st_file=AFROC_c2st_file.replace(".","")
    AFROC_c2st_file=AFROC_c2st_file+".pickle"

    with open(AFROC_c2st_file, "rb") as input_file:
        AFROC_c2st= pickle.load(input_file)
     
    AFROC_pool_file=results_directory+"/"+f"AFROC_Pool_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_pool_file=AFROC_pool_file.replace(".","")
    AFROC_pool_file=AFROC_pool_file+".pickle"

    with open(AFROC_pool_file, "rb") as input_file:
        AFROC_pool= pickle.load(input_file)

    AFROC_rulsif_file=results_directory+"/"+f"AFROC_Rulsif_experiment_{experiment}_alpha_{alpha}_N_{N_ref}"
    AFROC_rulsif_file=AFROC_rulsif_file.replace(".","")
    AFROC_rulsif_file=AFROC_rulsif_file+".pickle"

    with open(AFROC_rulsif_file, "rb") as input_file:
        AFROC_rulsif= pickle.load(input_file)

    AFROC_lstt_file=results_directory+"/"+f"AFROC_lstt_experiment_{experiment}_N_{N_ref}"
    AFROC_lstt_file=AFROC_lstt_file.replace(".","")
    AFROC_lstt_file=AFROC_lstt_file+".pickle"

    with open(AFROC_lstt_file, "rb") as input_file:
        AFROC_lstt= pickle.load(input_file)     
     
    AFROC_kliep_file=results_directory+"/"+f"AFROC_Kliep_experiment_{experiment}_N_{N_ref}"
    AFROC_kliep_file=AFROC_kliep_file.replace(".","")
    AFROC_kliep_file=AFROC_kliep_file+".pickle"

    with open(AFROC_kliep_file, "rb") as input_file:
        AFROC_kliep= pickle.load(input_file)   
     
    AFROC_MMD_median_file=results_directory+"/"+f"AFROC_MMD_median_experiment_{experiment}_N_{N_ref}"
    AFROC_MMD_median_file=AFROC_MMD_median_file.replace(".","")
    AFROC_MMD_median_file=AFROC_MMD_median_file+".pickle"

    with open(AFROC_MMD_median_file, "rb") as input_file:
        AFROC_MMD_median= pickle.load(input_file) 

    AFROC_MMD_max_file=results_directory+"/"+f"AFROC_MMD_max_experiment_{experiment}_N_{N_ref}"
    AFROC_MMD_max_file=AFROC_MMD_max_file.replace(".","")
    AFROC_MMD_max_file=AFROC_MMD_max_file+".pickle"

    with open(AFROC_MMD_max_file, "rb") as input_file:
        AFROC_MMD_max= pickle.load(input_file)          
    
    index_c2st=np.where(AFROC_c2st["FWR"]<=0.05)[0]
    index_pool=np.where(AFROC_pool["FWR"]<=0.05)[0]
    index_rulsif=np.where(AFROC_rulsif["FWR"]<=0.05)[0]
    index_lstt=np.where(AFROC_lstt["FWR"]<=0.05)[0]
    index_kliep=np.where(AFROC_kliep["FWR"]<=0.05)[0]
    index_mmd_median=np.where(AFROC_MMD_median["FWR"]<=0.05)[0]
    index_mmd_max=np.where(AFROC_MMD_max["FWR"]<=0.05)[0] 

    AFROC_c2st_FWR=np.hstack((0.05,AFROC_c2st["FWR"][index_c2st]))
    AFROC_c2st_TPR=np.hstack((np.max(AFROC_c2st["TPR"][index_c2st]),AFROC_c2st["TPR"][index_c2st]))
    AFROC_pool_FWR=np.hstack((0.05,AFROC_pool["FWR"][index_pool]))
    AFROC_pool_TPR=np.hstack((np.max(AFROC_pool["TPR"][index_pool]),AFROC_pool["TPR"][index_pool]))
    AFROC_rulsif_FWR=np.hstack((0.05,AFROC_rulsif["FWR"][index_rulsif]))
    AFROC_rulsif_TPR=np.hstack((np.max(AFROC_rulsif["TPR"][index_rulsif]),AFROC_rulsif["TPR"][index_rulsif]))
    AFROC_lstt_FWR=np.hstack((0.05,AFROC_lstt["FWR"][index_lstt]))
    AFROC_lstt_TPR=np.hstack((np.max(AFROC_lstt["TPR"][index_lstt]),AFROC_lstt["TPR"][index_lstt]))
    AFROC_kliep_FWR=np.hstack((0.05,AFROC_kliep["FWR"][index_kliep]))
    AFROC_kliep_TPR=np.hstack((np.max(AFROC_kliep["TPR"][index_kliep]),AFROC_kliep["TPR"][index_kliep]))
    AFROC_mmd_median_FWR=np.hstack((0.05,AFROC_MMD_median["FWR"][index_mmd_median]))
    AFROC_mmd_median_TPR=np.hstack((np.max(AFROC_MMD_median["TPR"][index_mmd_median]),AFROC_MMD_median["TPR"][index_mmd_median]))
    AFROC_mmd_max_FWR=np.hstack((0.05,AFROC_MMD_max["FWR"][index_mmd_max]))
    AFROC_mmd_max_TPR=np.hstack((np.max(AFROC_MMD_max["TPR"][index_mmd_max]),AFROC_MMD_max["TPR"][index_mmd_max]))

    fig = plt.figure(figsize=(12,10))
    plt.plot(AFROC_c2st_FWR,AFROC_c2st_TPR,label=r'C2ST $\alpha=0.1$',linestyle='solid',linewidth=5.0)
    plt.plot(AFROC_pool_FWR,AFROC_pool_TPR,label=r'POOL $\alpha=0.1$', linestyle='--',linewidth=5.0)
    plt.plot(AFROC_rulsif_FWR,AFROC_rulsif_TPR,label=r'RULSIF $\alpha=0.1$', linestyle='dotted',linewidth=5.0)
    plt.plot(AFROC_lstt_FWR,AFROC_lstt_TPR,label="LSTT",linestyle=(0,(5,10)),linewidth=5.0)
    plt.plot(AFROC_kliep_FWR,AFROC_kliep_TPR,label="KLIEP",linestyle="dashdot",linewidth=5.0)
    plt.plot(AFROC_mmd_median_FWR,AFROC_mmd_median_TPR,label="MMD-MEDIAN",linestyle=(0,(3,5,1,5)),linewidth=5.0)      
    plt.plot(AFROC_mmd_max_FWR,AFROC_mmd_max_TPR,label="MMD-MAX",linestyle=(0,(3,1,1,1)),linewidth=5.0)
    plt.ylim(-0.1, 1.1)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=100)
    plt.xlabel("FWER",fontsize=50)
    plt.ylabel("TPR",fontsize=50)
    plt.legend(loc='lower right') 
    plt.tight_layout()
    plt.savefig(results_directory+"/"+experiment+f"AFROC_N_{N_ref}_FW_FPR.pdf")
     
    AUC_FWR_c2st=metrics.auc(AFROC_c2st_FWR,AFROC_c2st_TPR)/0.05
    AUC_FWR_pool=metrics.auc(AFROC_pool_FWR,AFROC_pool_TPR)/0.05
    AUC_FWR_rulsif=metrics.auc(AFROC_rulsif_FWR,AFROC_rulsif_TPR)/0.05
    AUC_FWR_lstt=metrics.auc(AFROC_lstt_FWR,AFROC_lstt_TPR)/0.05
    AUC_FWR_kliep=metrics.auc(AFROC_kliep_FWR,AFROC_kliep_TPR)/0.05
    AUC_FWR_mmd_median=metrics.auc(AFROC_mmd_median_FWR,AFROC_mmd_median_TPR)/0.05
    AUC_FWR_mmd_max=metrics.auc(AFROC_mmd_max_FWR,AFROC_mmd_max_TPR)/0.05

    AUC_FWR=np.vstack((AUC_FWR_c2st,AUC_FWR_pool,AUC_FWR_rulsif,AUC_FWR_lstt,AUC_FWR_kliep,AUC_FWR_mmd_median,AUC_FWR_mmd_max))
    AUC_FWR=pd.DataFrame(AUC_FWR,index=["C2ST","POOL","RULSIF","LSTT","KLIEP","MMD-median","MMD-max"])
    AUC_FWR.to_csv(results_directory+"/"+experiment+f"AUC_FWR_N_{N_ref}.csv") 
         
    FPR_c2st=AFROC_c2st["FPR"]
    TPR_c2st=AFROC_c2st["TPR"]
    FPR_pool=AFROC_pool["FPR"]
    TPR_pool=AFROC_pool["TPR"]
    FPR_rulsif=AFROC_rulsif["FPR"]
    TPR_rulsif=AFROC_rulsif["TPR"]
    FPR_lstt=AFROC_lstt["FPR"]
    TPR_lstt=AFROC_lstt["TPR"]
    FPR_kliep=AFROC_kliep["FPR"]
    TPR_kliep=AFROC_kliep["TPR"]
    FPR_mmd_median=AFROC_MMD_median["FPR"]
    TPR_mmd_median=AFROC_MMD_median["TPR"]
    FPR_mmd_max=AFROC_MMD_max["FPR"]
    TPR_mmd_max=AFROC_MMD_max["TPR"]

    fig = plt.figure(figsize=(12,10))
    plt.plot(FPR_c2st,TPR_c2st,label=r'C2ST $\alpha=0.1$',linestyle='solid')
    plt.plot(FPR_pool,TPR_pool,label=r'POOL $\alpha=0.1$', linestyle='--')
    plt.plot(FPR_rulsif,TPR_rulsif,label=r'RULSIF $\alpha=0.1$', linestyle='dotted')
    plt.plot(FPR_lstt,TPR_lstt,label="ULSIF",linestyle=(0,(5,10)))
    plt.plot(FPR_kliep,TPR_kliep,label="KLIEP",linestyle="dashdot")
    plt.plot(FPR_mmd_median,TPR_mmd_median,label="MMD-MEDIAN",linestyle=(0,(3,5,1,5)))      
    plt.plot(FPR_mmd_max,TPR_mmd_max,label="MMD-MAX",linestyle=(0,(3,1,1,1))) 
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("FPR",fontsize=50)
    plt.ylabel("TPR",fontsize=50)
    plt.savefig(results_directory+"/"+experiment+f"ROC_N_{N_ref}.pdf")

    AUC_c2st=metrics.auc(FPR_c2st,TPR_c2st)
    AUC_pool=metrics.auc(FPR_pool,TPR_pool)
    AUC_rulsif=metrics.auc(FPR_rulsif,TPR_rulsif)
    AUC_lstt=metrics.auc(FPR_lstt,TPR_lstt)
    AUC_kliep=metrics.auc(FPR_kliep,TPR_kliep)
    AUC_mmd_median=metrics.auc(FPR_mmd_median,TPR_mmd_median)
    AUC_mmd_max=metrics.auc(FPR_mmd_max,TPR_mmd_max)

    AUC=np.vstack((AUC_c2st,AUC_pool,AUC_rulsif,AUC_lstt,AUC_kliep,AUC_mmd_median,AUC_mmd_max))
    AUC=pd.DataFrame(AUC,index=["C2ST","POOL","RULSIF","LSTT","KLIEP","MMD-median","MMD-max"])
    AUC.to_csv(results_directory+"/"+experiment+f"AUC_N_{N_ref}.csv")   
    
    
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory",type=str) #### Dictionary where the results will be saved
    parser.add_argument("--experiment",type=str) #### The name of the model to be run
    parser.add_argument("--N",type=int) #### The experiment to be run
    
    args=parser.parse_args()
    
    results_directory=args.results_directory
    experiment=args.experiment
    N=args.N

    plot_AFROC_ROC_synthetic_experiments(results_directory,experiment,N)


     
     
     
     
     
     
     
     
     
          