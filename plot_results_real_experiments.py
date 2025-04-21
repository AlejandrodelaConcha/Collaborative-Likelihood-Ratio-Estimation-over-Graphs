# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  plot_results_real_experiments.py 
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2024-01-15              
# This version:     2024-01-15
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): To implement the functions required to produce visualizations similar to those appearing in the paper. 
#               This includes maps showing the epicenter and the reacting stations, as well as time series displaying 
#               the time periods of major reactions.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: geopandas, contextily, matplotlib, xyzservices, pandas, numpy, pygsp, scipy, matplotlib, shapely, Models.aux_functions.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Seismics, New Zealand, epicenter
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Comments: It is important that:
# 1. The preprocessing of the event of interest has been completed by running preprocess_seismic.
# 2. The multiple hypothesis testing results for all the methods being compared are available.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import pickle
import pandas as pd
from pygsp import graphs

from Visualization.plot_seismics import *
from Models.aux_functions import *

def main(data_directory,results_directory,event_id,p_value):
    ### This code runs all the methods for detecting the location and data stamp which are statistically significant.
    ## Input:
    # data_directory: location where the datasets are stored.
    # results_directory: the folder where the results will be stored.
    # event_id: the seismic event ID.
    # p_value: the level of significance to spot statisticall significant values (recommended=0.01, or 0.05)
    
    
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
    G_seismic.plot()

    network_stations=adjacency_network.index

    complete_data_ref=[data[station][1:1001] for station in network_stations]
    complete_data_test=[data[station][1001:] for station in network_stations]
         


    with open(results_directory+'/c2st_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_grulsif=pickle.load(f)

    with open(results_directory+'/Pool_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_pool=pickle.load(f)
        
    with open(results_directory+'/rulsif_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_rulsif=pickle.load(f)
        
    with open(results_directory+'/lstt_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_ulsif=pickle.load(f)
        
    with open(results_directory+'/kliep_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_kliep=pickle.load(f)
           
    with open(results_directory+'/mmd_median_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_mmd_median=pickle.load(f)
        
    with open(results_directory+'/mmd_max_New_Zealand'+event_id+'_scores.pickle', 'rb') as f:
        score_mmd_max=pickle.load(f)

    n_nodes=G_seismic.W.shape[0]
    n_times=10  
        
    p_1_grulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_grulsif["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_1_grulsif[i]=np.sum(max_scores>=score_grulsif["score_pq"][i])/1000
        
    p_2_grulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_grulsif["scores_qp"]]
    for i in range(n_nodes*n_times):
        p_2_grulsif[i]=np.sum(max_scores>=score_grulsif["score_qp"][i])/1000
        
        
    p_1_pool=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_pool["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_1_pool[i]=np.sum(max_scores>=score_pool["score_pq"][i])/1000
        
    p_2_pool=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_pool["scores_qp"]]
    for i in range(n_nodes*n_times):
        p_2_pool[i]=np.sum(max_scores>=score_pool["score_qp"][i])/1000

    p_1_rulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_rulsif["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_1_rulsif[i]=np.sum(max_scores>=score_rulsif["score_pq"][i])/1000
        
    p_2_rulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_rulsif["scores_qp"]]
    for i in range(n_nodes*n_times):
        p_2_rulsif[i]=np.sum(max_scores>=score_rulsif["score_qp"][i])/1000
        
        
    p_1_ulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_ulsif["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_1_ulsif[i]=np.sum(max_scores>=score_ulsif["score_pq"][i])/1000
        
    p_2_ulsif=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_ulsif["scores_qp"]]
    for i in range(n_nodes*n_times):
        p_2_ulsif[i]=np.sum(max_scores>=score_ulsif["score_qp"][i])/1000
            
    p_1_kliep=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_kliep["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_1_kliep[i]=np.sum(max_scores>=score_kliep["score_pq"][i])/1000    
        
    p_2_kliep=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_kliep["scores_qp"]]
    for i in range(n_nodes*n_times):
        p_2_kliep[i]=np.sum(max_scores>=score_kliep["score_qp"][i])/1000    
         
    p_mmd_median=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_mmd_median["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_mmd_median[i]=np.sum(max_scores>=score_mmd_median["score_pq"][i])/1000       
        
    p_mmd_max=np.zeros(n_nodes*n_times)
    max_scores=[np.max(score) for score in score_mmd_max["scores_pq"]]
    for i in range(n_nodes*n_times):
        p_mmd_max[i]=np.sum(max_scores>=score_mmd_max["score_pq"][i])/1000  
        
    sample_size_node=100
    n_times=10
    n_nodes=len(adjacency_network)
    W_1=transform_matrix_totime(G_seismic.W.tocoo(),n_times=n_times)

    file_name=results_directory+"/c2st_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_1_grulsif,p_2_grulsif,p_value=p_value,min_size_cluster="max")

    file_name=results_directory+"/Pool_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_1_pool,p_2_pool,p_value=p_value,min_size_cluster="max")

    file_name=results_directory+"/Kliep_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_1_kliep,p_2_kliep,p_value=p_value,min_size_cluster="max")

    file_name=results_directory+"/Rulsif_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_1_rulsif,p_2_rulsif,p_value=p_value,min_size_cluster='max')

    file_name=results_directory+"/lstt_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_1_ulsif,p_2_ulsif,p_value=p_value,min_size_cluster='max')

    file_name=results_directory+"/MMD_median_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_mmd_median,p_value=p_value,min_size_cluster='max')

    file_name=results_directory+"/MMD_max_New_Zealand"+event_id
    plot_results_seismic(file_name,complete_data_ref,complete_data_test,G_seismic,epicenter,sample_size_node,n_times,n_nodes,p_mmd_max,p_value=p_value,min_size_cluster='max')



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--data_directory",type=str) #### Dictionary where the datasets are stored
    parser.add_argument("--results_directory",type=str) #### Dictionary where the datasets are stored
    parser.add_argument("--eventid",type=str) #### The event that will be analyzed. This identifier is available in the GEONET website. 
    parser.add_argument("--p_value",type=float)  ############## The level of FWER control. 

    args=parser.parse_args()
    
    data_directory=args.data_directory
    results_directory=args.results_directory
    event_id=args.eventid
    p_value=args.p_value
    
    main(data_directory,results_directory,event_id,p_value)
    
