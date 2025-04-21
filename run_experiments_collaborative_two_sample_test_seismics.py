# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  run_experiments_collaborative_two_sample_test_seismics
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-28              
# This version:     2025-02-28
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to replicate the experiments presented in the paper, specifically for the seismic datasets.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: pickle, argparse, pandas, pygsp, Models.collaborative_two_sample_test, Models.two_sample_test_univariate_models
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Collaborative TST, Pool, RULSIF, LSTT, MMD, C2ST, KLIEP, seismic
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Comments: It is important that: 
# 1. The preprocessing of the event of interest must be done first (run preprocess_seismic_two_sample_test.py).
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import pickle
import pandas as pd
from pygsp import graphs

from Models.collaborative_two_sample_test import *
from Models.two_sample_test_univariate_models import *

def main(data_directory,results_directory,event_id):
    ### This code runs all the methods for detecting the location and data stamp which are statistically significant.
    ## Input:
    # data_directory: location where the datasets are stored.
    # results_directory: the folder where the results will be stored.
    # event_id: the seismic event ID.
 
    
    data_file=data_directory+'/'+event_id+'_data.pickle'
    network_file=data_directory+"/New_Zealand_Network_"+event_id+".csv"
    coordinates_file=data_directory+"/New_Zealand_coordinates"+event_id+".csv"

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    epicenter=data["epicenter"]
    data=data["waveforms"]

        
    adjacency_network=pd.read_csv(network_file,index_col=0)  
    coordinates_network=pd.read_csv(coordinates_file,index_col=0)  

    G_seismic=graphs.Graph(adjacency_network)

    G_seismic.set_coordinates(coordinates_network)

    network_stations=adjacency_network.index
        
    N_min=np.min([len(data[station]) for station in network_stations]) 
    N_min=np.min((N_min,2000))
    
    complete_data_ref=[data[station][:1000] for station in network_stations]
    complete_data_test=[data[station][1000:N_min] for station in network_stations]
    
    data_ref=[d.reshape((10,100,3)) for d in complete_data_ref]
    data_test=[d.reshape((10,100,3)) for d in complete_data_test]

    print("Running C2ST")
    alpha=0.1
    threshold_coherence=0.1
    c2st=C2ST(G_seismic.W,data_ref,data_test,threshold_coherence=threshold_coherence,alpha=alpha,verbose=True,tol=1e-2, time=True)
    p_1_c2st,p_2_c2st=c2st.get_pivalues(n_rounds=1000)

    score_c2st={"score_pq":c2st.score_pq,"score_qp":c2st.score_qp,"scores_pq":c2st.PD_1s,"scores_qp":c2st.PD_2s}
    with open(results_directory+'/c2st_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_c2st, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    del c2st
    del p_1_c2st
    del p_2_c2st
    del score_c2st 
        
        
    print("Running Pool")
    pool_tst=Pool_two_sample_test(data_ref, data_test, threshold_coherence=threshold_coherence,  alpha=alpha, tol=1e-2, verbose=False, time=True)
    p_1_pool,p_2_pool=pool_tst.get_pivalues(n_rounds=1000)
    
    score_pool={"score_pq":pool_tst.score_pq,"score_qp":pool_tst.score_qp,"scores_pq":pool_tst.PD_1s,"scores_qp":pool_tst.PD_2s}
    
    with open(results_directory+'/Pool_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    
    del pool_tst
    del p_1_pool
    del p_2_pool
    del score_pool     


    print("Running RULSIF")
    alpha=0.1
    rulsif=RULSIF_two_sample_test(data_ref,data_test,alpha=alpha,time=True)
    p_1_rulsif,p_2_rulsif=rulsif.get_pivalues(n_rounds=1000)
   
    score_rulsif={"score_pq":rulsif.score_pq,"score_qp":rulsif.score_qp,"scores_pq":rulsif.PD_1s,"scores_qp":rulsif.PD_2s}
    with open(results_directory+'/rulsif_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_rulsif, handle, protocol=pickle.HIGHEST_PROTOCOL)    
           
    del rulsif
    del p_1_rulsif
    del p_2_rulsif
    del score_rulsif        

    print("Running LSTT")
    lstt=LSTT(data_ref,data_test,time=True)
    p_1_lstt,p_2_lstt=lstt.get_pivalues(n_rounds=1000)
    
    score_lstt={"score_pq":lstt.score_pq,"score_qp":lstt.score_qp,"scores_pq":lstt.PD_1s,"scores_qp":lstt.PD_2s}
    with open(results_directory+'/lstt_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_lstt, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    del lstt
    del p_1_lstt
    del p_2_lstt
    del score_lstt          

    print("Running KLIEP")
    kliep=KLIEP_two_sample_test(data_ref,data_test,lr=1e-4,tol=1e-3,time=True)
    p_1_kliep,p_2_kliep=kliep.get_pivalues(n_rounds=1000)
      
    score_kliep={"score_pq":kliep.score_pq,"score_qp":kliep.score_qp,"scores_pq":kliep.KL_1s,"scores_qp":kliep.KL_2s}
    with open(results_directory+'/kliep_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_kliep, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del kliep
    del p_1_kliep
    del p_2_kliep
    del score_kliep 
    
    print("Running MMD median")
    mmd_median=MMD_two_sample_test(data_ref,data_test,estimate_sigma=False,time=True)
    p_mmd_median=mmd_median.get_pivalues(n_rounds=1000) 
    
    score_mmd_median={"score_pq":mmd_median.scores,"scores_pq":mmd_median.MMD_s}
    with open(results_directory+'/mmd_median_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_mmd_median, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    del mmd_median
    del p_mmd_median
    del score_mmd_median

    print("Running MMD max")
    mmd_max=MMD_two_sample_test(data_ref,data_test,estimate_sigma=True,time=True)
    p_mmd_max=mmd_max.get_pivalues(n_rounds=1000)

    score_mmd_max={"score_pq":mmd_max.scores,"scores_pq":mmd_max.MMD_s}
    with open(results_directory+'/mmd_max_New_Zealand'+event_id+'_scores.pickle', 'wb') as handle:
        pickle.dump(score_mmd_max, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    del mmd_max
    del  p_mmd_max
    del score_mmd_max

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--data_directory",type=str) #### Dictionary where the datasets are stored
    parser.add_argument("--results_directory",type=str) #### Dictionary where the datasets are stored
    parser.add_argument("--eventid",type=str) #### The event that will be analyzed. This identifier is available in the GEONET website. 
   
    args=parser.parse_args()
    
    data_directory=args.data_directory
    results_directory=args.results_directory
    event_id=args.eventid
    
    main(data_directory,results_directory,event_id)
    
