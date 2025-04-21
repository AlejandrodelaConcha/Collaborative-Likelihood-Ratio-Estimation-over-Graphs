# -----------------------------------------------------------------------------------------------------------------
# Title:  run_experiments_collaborative_LRE_seismic_dataset
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-26             
# This version:     2025-02-26
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to compare the performance of collaborative likelihood-ratio estimation 
#               using the seismic dataset provided by the GEONET project.
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# obspy, pandas, numpy, matplotlib, statsmodels, sklearn, scipy, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GEONET, seismics, k-nearest neighbor
# -----------------------------------------------------------------------------------------------------------------
# This script should be run after the inputs for the specified event_id have been generated 
# using the script preprocess_seismic.


import argparse
from Evaluation.evaluation_metrics import *
import pickle
import pandas as pd
from pygsp import graphs

from Models.likelihood_ratio_collaborative import *
from Models.likelihood_ratio_univariate import *
from Evaluation.evaluation_metrics import *

def main(data_directory,results_directory,event_id,alpha=0.1,threshold_coherence=0.3):
    
    ## This function estimates the likelihood ratio using each of the methods presented in the paper. 

    ### Input
    # data_directory: the folder where the clean data is stored.
    # results_directory: the folder where the MSE of the methods is compared.
    # event_id: the seismic event being analyzed (accessible at https://www.geonet.org.nz/).
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio (used in GRULSIF, POOL, and RULSIF).
    # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
    #                      When the kernel is normal, this parameter should be between 0 and 1.
    #                      The closer it is to 1, the larger the dictionary and the slower the training.

    #### Open datasets
    
    data_file=data_directory+'/'+event_id+'_data.pickle'
    network_file=data_directory+"/New_Zealand_Network_"+event_id+".csv"
    coordinates_file=data_directory+"/New_Zealand_coordinates"+event_id+".csv" 
    
    ### Divide data
    
    adjacency_network=pd.read_csv(network_file,index_col=0)  
    network_stations=adjacency_network.index
    G_seismic=graphs.Graph(adjacency_network)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        
    data=data["waveforms"] 
    data_ref=[data[station][1:601] for station in network_stations]
    data_test=[data[station][601:1200] for station in network_stations]  
    
    N_ref=len(data_ref[0])
    N_test=len(data_test[0])
    n_nodes=len(data_ref)
      
    ### Prepare training and validation dataset
    np.random.seed(0)
    aux_index_ref=np.arange(N_ref)
    np.random.shuffle(aux_index_ref)
    aux_index_test=np.arange(N_test)
    np.random.shuffle(aux_index_test)
    ref_index_validation=aux_index_ref[:int(N_ref/5)]
    test_index_validation=aux_index_test[:int(N_test/5)]
    ref_index_train=aux_index_ref[int(N_ref/5):]
    test_index_train=aux_index_test[int(N_test/5):]  
    
    data_ref_train=[d[ref_index_train] for d in data_ref]
    data_test_train=[d[test_index_train] for d in data_test]
    data_ref_validation=[d[ref_index_validation] for d in data_ref]
    data_test_validation=[d[test_index_validation] for d in data_test]
    
    ############ Likelihood ratio modes
    
    real_likelihoods=[lambda x: np.ones(len(x)) for i in range(n_nodes)]

    
    ##### grulsif
    grulsif=GRULSIF(G_seismic.W,data_ref_train,data_test_train,alpha=alpha,verbose=False,tol=1e-3,threshold_coherence=threshold_coherence)
    theta_grulsif=grulsif.fit(data_ref_train,data_test_train,tol=1e-3)
    L2_grulsif=L2_error(data_ref_validation,data_test_validation,alpha,real_likelihoods,theta_grulsif,grulsif)
    
    model_parameters_grulsif={"kernel":grulsif.kernel,"lamb":grulsif.lamb,
                  "gamma":grulsif.gamma,"G":grulsif.W}
    
    file_name=results_directory+"/"+f"Grulsif_alpha_{alpha}_event_id_{event_id}"    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(L2_grulsif, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_LRE={"theta":theta_grulsif,"LREmodel_":model_parameters_grulsif}
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    ##### pool 
    pool=Pool(data_ref_train,data_test_train,alpha=alpha,verbose=False,tol=1e-3,threshold_coherence=threshold_coherence)
    theta_pool=pool.fit(data_ref_train,data_test_train,tol=1e-3)
    L2_pool=L2_error(data_ref_validation,data_test_validation,alpha,real_likelihoods,theta_pool,pool)
    
    model_parameters_pool={"kernel":pool.kernel,"gamma":pool.gamma}
    
    file_name=results_directory+"/"+f"Pool_alpha_{alpha}_event_id_{event_id}"    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(L2_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_LRE={"theta":theta_pool,"LREmodel_":model_parameters_pool}
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #### rulsif 
    
    rulsif=RULSIF_nodes(data_ref_train,data_test_train,alpha=alpha)
    theta_rulsif=rulsif.fit(data_ref_train,data_test_train)
    L2_rulsif=L2_error(data_ref_validation,data_test_validation,alpha,real_likelihoods,theta_rulsif,rulsif)
    
    kernels_=[rulsif.RULSIF_models[i].kernel for i in range(n_nodes)]
    gammas_=[rulsif.RULSIF_models[i].gamma for i in range(n_nodes)]

    model_parameters_rulsif={"kernel":kernels_,"gamma":gammas_}  
    
    file_name=results_directory+"/"+f"Rulsif_alpha_{alpha}_event_id_{event_id}"    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(L2_rulsif, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_LRE={"theta":theta_rulsif,"LREmodel_":model_parameters_rulsif}
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #### ulsif 
    
    ulsif=ULSIF_nodes(data_ref_train,data_test_train)
    theta_ulsif=ulsif.fit(data_ref_train,data_test_train)
    L2_ulsif=L2_error(data_ref_validation,data_test_validation,alpha=0.0,real_likelihoods=real_likelihoods,
                      theta=theta_ulsif,model_likelihood_ratios=ulsif)

    
    kernels_=[ulsif.ULSIF_models[i].kernel for i in range(n_nodes)]
    gammas_=[ulsif.ULSIF_models[i].gamma for i in range(n_nodes)]

    model_parameters_ulsif={"kernel":kernels_,"gamma":gammas_}  
    
    file_name=results_directory+"/"+f"Ulsif_event_id_{event_id}"    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(L2_ulsif, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_LRE={"theta":theta_ulsif,"LREmodel_":model_parameters_ulsif}
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    kliep=KLIEP_nodes(data_ref_train,data_test_train)
    theta_kliep=kliep.fit(data_ref_train,data_test_train)
    L2_kliep=L2_error(data_ref_validation,data_test_validation,alpha=0.0,real_likelihoods=real_likelihoods,
                      theta=theta_kliep,model_likelihood_ratios=kliep)

    
    kernels_=[kliep.KLIEP_models[i].kernel for i in range(n_nodes)]

    model_parameters_kliep={"kernel":kernels_}  
    
    file_name=results_directory+"/"+f"Kliep_event_id_{event_id}"    
    file_name=file_name.replace(".","")
    
    file_name_errors=file_name+"_errors"
    file_name_LRE=file_name+"_LRE_results"
    
    with open(file_name_errors+".pickle", 'wb') as handle:
        pickle.dump(L2_kliep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_LRE={"theta":theta_kliep,"LREmodel_":model_parameters_kliep}
    
    with open( file_name_LRE+".pickle", 'wb') as handle:
        pickle.dump(results_LRE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    L2_errors=np.array([np.mean(L2_grulsif),np.mean(L2_pool),np.mean(L2_rulsif),np.mean(L2_ulsif),np.mean(L2_kliep)])
    
    L2_errors=pd.DataFrame(L2_errors,index=["grulsif","pool","rulsif","ulsif","kliep"])
    
    L2_errors.to_csv(results_directory+"/"+f"L2_errors_event_id_{event_id}.csv",header=False)
    
    
        
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--data_directory") #### Directory where the data is stored
    parser.add_argument("--results_directory") #### Directory where the results will be saved
    parser.add_argument("--alpha",default=0.0,type=float) #### the regularization parameter for the likelihood ratio 
    parser.add_argument("--threshold_coherence",default=0.1,type=float) #### the regularization parameter for the likelihood ratio 
    parser.add_argument("--event_id",type=str) #### The name of the model to be run
   
    
    args=parser.parse_args()
    
    data_directory=args.data_directory
    results_directory=args.results_directory
    event_id=args.event_id 
    alpha=args.alpha
    threshold_coherence=args.threshold_coherence
    
    main(data_directory,results_directory,event_id,alpha,threshold_coherence)
    
    
    
    
    
    
  
    
    
    
    
    
    
    