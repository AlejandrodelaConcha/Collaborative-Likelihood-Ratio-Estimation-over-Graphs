# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  evaluation_metrics
# Author(s):  
# Initial version:  2024-01-15
# Last modified:    2025-02-25              
# This version:     2025-02-25 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to replicate the validation metrics presented in the paper.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: pickle, argparse, Experiments
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: L2 distance, experiments, evaluation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np 
from Experiments.experiments_LRE import generate_experiment_1_LRE,generate_experiment_2_LRE,generate_experiment_3_LRE,generate_experiment_4_LRE,real_relative_likelihood_ratio_C_1,real_relative_likelihood_ratio_C_2,real_relative_likelihood_ratio_C_3_C_4,r_mul_normal_C1,r_mul_normal_C2,r_mul_normal_C3,r_mul_normal_C4,r_mul_normal_non_C,r_mul_normal_C,r_mul_normal_non_C_2B,r_normal_mixture,real_relative_likelihood_ratio_grid,real_relative_likelihood_ratio_highdimension                                                                                         

def L2_error(data_ref_validation,data_test_validation,alpha,real_likelihoods,theta,model_likelihood_ratios):
    
    ######## This function estimates the convergence in L2 of different algorithms. 

    ###### Before running this function, the results of the experiments should be produced. 

    ########### Input 
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # alpha: parameter to upper-bound the likelihood ratio.
    # real_likelihood: function to compute the exact relative likelihood ratio.
    # model_likelihood_ratios: the estimated relative likelihood ratios.

    ########### Output 
    # L_2_distance: the L2 distance between the estimated likelihood ratio and the real likelihood ratio.


    L_2_distance=np.zeros(len(theta))
    
   
    for i in range(0,len(theta)):
        r_real_ref=real_likelihoods[i](data_ref_validation[i])
        r_real_test=real_likelihoods[i](data_test_validation[i])
        r_v_ref=model_likelihood_ratios.r_v(i,theta,data_ref_validation[i])
        r_v_test=model_likelihood_ratios.r_v(i,theta,data_test_validation[i])
    
        if alpha>0:
            L_2_distance[i]=(1-alpha)*np.mean((r_real_ref-r_v_ref)**2)
            L_2_distance[i]+=alpha*np.mean((r_real_test-r_v_test)**2)
        else:
            L_2_distance[i]=np.mean((r_real_ref-r_v_ref)**2)
              
    return L_2_distance


################################### Evaluation LRE 

def evaluate_models(experiment,n_nodes,alpha,theta,model_likelihood_ratios): 

    ######## This function generates the convergence  of different algorithms 
    ######## under the scenarios described in the main text.

    ###### Before running this function, the results of the experiments should be produced. 

    ########### Input 
    # experiment: the experiment being studied: '1A', '1B', '2A', '2B', '2C'.
    # n_nodes: the number of nodes used in the experiment.
    # alpha: the regularization parameter of the likelihood ratio.
    # theta: the parameter related to the likelihood ratios.
    # model_likelihood_ratios: the estimated likelihood ratio related to the model being tested.

    ########### Output 
    # errors: the L2 distance between the estimated likelihood ratio and the real likelihood ratio.


    N_ref=10000
    N_test=10000

    if experiment=="1A":
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test)
        
        n_clusters=4
        sample_size=int(n_nodes/n_clusters)

        alpha=0.1
        real_likelihoods=[]
        for i in range(len(data_ref_validation)):
            if i in range(sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_1(x,alpha=alpha))
            elif i in range(sample_size,3*sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_3_C_4(x,alpha=alpha))
            elif i in range(3*sample_size,4*sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_2(x,alpha=alpha))  
                
    elif experiment=="1B": 
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test)
        
        n_clusters=4
        sample_size=int(n_nodes/n_clusters)

        real_likelihoods=[]
        for i in range(len(data_ref_validation)):
            if i in range(sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C1(x,alpha=alpha))
            elif i in range(sample_size,2*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C2(x,alpha=alpha))
            elif i in range(2*sample_size,3*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C3(x,alpha=alpha))
            elif i in range(3*sample_size,4*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C4(x,alpha=alpha)) 
                
                
            
    elif experiment=="2A":
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))

        A=np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2)],[np.sin(-np.pi/2),np.cos(-np.pi/2)]])
        n_nodes=int(N_1*N_2)
        coordinates=np.zeros(shape=(n_nodes,2))
        for i in range(n_nodes):
            n_row=int(i/N_2)
            n_col=i % N_2
            coordinates[i,:]=A.dot(np.array([n_row,n_col])-np.array([N_1/2,N_2/2]))
            
        coordinates[:,0]/=np.max(np.abs(coordinates[:,0]))
        coordinates[:,1]/=np.max(np.abs(coordinates[:,1]))
        coordinates*=2.0
              
        real_likelihoods=[]
        for i in range(n_nodes):
            real_likelihoods.append(lambda x:real_relative_likelihood_ratio_grid(x,coordinates[i,:],alpha=alpha))
            
    elif experiment=="2B":
        d=2
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))
        n_nodes=int(N_1*N_2)
        
        cov_blocks=np.eye(int(2*d))
        for i in range(int(d)):
            cov_blocks[2*i,2*i+1]=0.8
            cov_blocks[2*i+1,2*i]=0.8
            
        real_likelihoods=[]
        for i in range(n_nodes):
            if i in affected_nodes: 
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,cov_blocks,d=d,alpha=alpha))
            else:
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,np.eye(int(2*d)),d=d,alpha=alpha))
         
            
    elif experiment=="2C":
        d=10
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))
        n_nodes=int(N_1*N_2)
         
        cov_blocks=np.eye(int(2*d))
        for i in range(int(d)):
            cov_blocks[2*i,2*i+1]=0.8
            cov_blocks[2*i+1,2*i]=0.8
            
        real_likelihoods=[]
        for i in range(n_nodes):
            if i in affected_nodes: 
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,cov_blocks,d=d,alpha=alpha))
            else:
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,np.eye(int(2*d)),d=d,alpha=alpha))
                
        
    
    errors=L2_error(data_ref_validation,data_test_validation,alpha,real_likelihoods,theta,model_likelihood_ratios)
    
    return errors
            
    

def estimate_real_f_div(experiment,n_nodes,alpha=0.1,d=50,type="PE"): 
    
    ### This function estimates the real f-divergence of the models described in the paper. 
    ### An f-divergence estimate is produced for each of the nodes. 

    ### Input
    # experiment: the experiment to be used: "1A", "1B", "2A", "2B","2C".
    # n_nodes: the number of nodes in the graph.
    # alpha: parameter to upper-bound the likelihood ratio.
    # d: the dimension of experiment 2B and 2C
    # type: the type of divergence to be estimated: "PE" or "KL".

    ### Output 
    # f_div: the f-divergence estimate associated with each of the nodes.

    
    f_div=np.zeros(n_nodes)
    
    N_ref=int(1e5)
    N_test=int(1e5)

    if experiment=="1A":
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_1_LRE(n_nodes,N_ref,N_test)
        
        n_clusters=4
        sample_size=int(n_nodes/n_clusters)

        real_likelihoods=[]
        for i in range(len(data_ref_validation)):
            if i in range(sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_1(x,alpha=alpha))
            elif i in range(sample_size,3*sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_3_C_4(x,alpha=alpha))
            elif i in range(3*sample_size,4*sample_size):
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_C_2(x,alpha=alpha))  
        
    
                
    elif experiment=="1B": 
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_2_LRE(n_nodes,N_ref,N_test)

        n_clusters=4
        sample_size=int(n_nodes/n_clusters)

        real_likelihoods=[]
        for i in range(len(data_ref_validation)):
            if i in range(sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C1(x,alpha=alpha))
            elif i in range(sample_size,2*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C2(x,alpha=alpha))
            elif i in range(2*sample_size,3*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C3(x,alpha=alpha))
            elif i in range(3*sample_size,4*sample_size):
                real_likelihoods.append(lambda x:r_mul_normal_C4(x,alpha=alpha)) 
                           
    elif experiment=="2A":
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_3_LRE(n_nodes,N_ref,N_test)
    
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))

        A=np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2)],[np.sin(-np.pi/2),np.cos(-np.pi/2)]])
        n_nodes=int(N_1*N_2)
        coordinates=np.zeros(shape=(n_nodes,2))
        for i in range(n_nodes):
            n_row=int(i/N_2)
            n_col=i % N_2
            coordinates[i,:]=A.dot(np.array([n_row,n_col])-np.array([N_1/2,N_2/2]))
            
        coordinates[:,0]/=np.max(np.abs(coordinates[:,0]))
        coordinates[:,1]/=np.max(np.abs(coordinates[:,1]))
        coordinates*=2.0
          
        real_likelihoods=[]
        for i in range(n_nodes):
            real_likelihoods.append(lambda x:real_relative_likelihood_ratio_grid(x,coordinates[i,:],alpha=alpha))
 
    elif experiment=="2B":
        d=2
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))
        n_nodes=int(N_1*N_2)
        
        cov_blocks=np.eye(int(2*d))
        for i in range(int(d)):
            cov_blocks[2*i,2*i+1]=0.8
            cov_blocks[2*i+1,2*i]=0.8
 
        real_likelihoods=[]
        for i in range(n_nodes):
            if i in affected_nodes: 
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,cov_blocks,d=d,alpha=alpha))
            else:
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,np.eye(int(2*d)),d=d,alpha=alpha))

    elif experiment=="2C":
        d=10
        _,data_ref_validation,data_test_validation,affected_nodes=generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=d)
        
        N_1=int(np.sqrt(n_nodes))
        N_2=int(np.ceil(n_nodes/N_1))
        n_nodes=int(N_1*N_2)
        
        cov_blocks=np.eye(int(2*d))
        for i in range(int(d)):
            cov_blocks[2*i,2*i+1]=0.8
            cov_blocks[2*i+1,2*i]=0.8
 
        real_likelihoods=[]
        for i in range(n_nodes):
            if i in affected_nodes: 
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,cov_blocks,d=d,alpha=alpha))
            else:
                real_likelihoods.append(lambda x:real_relative_likelihood_ratio_highdimension(x,np.eye(int(2*d)),d=d,alpha=alpha))
    
            
    if type=="PE":
        
        for i in range(len(data_ref_validation)):
            r_real_ref=real_likelihoods[i](data_ref_validation[i])
            r_real_test=real_likelihoods[i](data_test_validation[i])
            if alpha>0:
                f_div[i]=(1-alpha)*np.mean(r_real_ref**2)
                f_div[i]+=alpha*np.mean((r_real_test**2))
                f_div[i]*=-0.5
                f_div[i]+=np.mean(r_real_test)  
                f_div[i]-=0.5
            else:
                f_div[i]=np.mean(r_real_ref**2)
                f_div[i]*=-0.5
                f_div[i]+=np.mean(r_real_test)  
                f_div[i]-=0.5
        
    elif type=="KL":
        for i in range(len(data_ref_validation)):
            r_real_test=real_likelihoods[i](data_test_validation[i])
            f_div[i]=np.mean(np.log(r_real_test+1e-6))
    return f_div

    
def f_div(theta,kernel,data_ref,data_test,alpha=0.1,model="grulsif"):
    
    #### Given an LRE model, this function estimates the associated f-divergence.

    ### Input
    # theta: the parameter related to the likelihood ratios.
    # kernel: the kernel function to be used.
    # data_ref: data from the distribution x ~ P.
    # data_test: data from the distribution x ~ Q.
    # model: LRE model to be used; one of ["grulsif", "pool", "rulsif", "ulsif", "kliep"].

    ### Output 
    # score: the f-divergence estimate associated with each of the nodes.

    
    phi_test=[]
    phi_ref=[] 
    n_nodes=len(theta)
    score=np.zeros(n_nodes)
    
    if model in ["rulsif","ulsif"]:

        for i in range(n_nodes):
            phi_test.append(kernel[i].k_V(data_test[i]))
            phi_ref.append(kernel[i].k_V(data_ref[i]))
        
        N_ref=len(phi_ref[0])
        N_test=len(phi_test[0])
        h_test =[[] for n in range(n_nodes)]
        H_ref = [[] for n in range(n_nodes)]
        H_test=[[] for n in range(n_nodes)]
        
        for i in range(n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H_ref[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H_test[i]=np.einsum('ji,j...',  phi_test[i],phi_test[i])
            H_ref[i]/=N_ref 
            H_test[i]/=N_test
        
        if model in ["ulsif"]:
            alpha=0.0
      
                
        for i in range(n_nodes):
             score[i]=(theta[i].T).dot(H_test[i]).dot(theta[i])*alpha
             score[i]+=(theta[i].T).dot(H_ref[i]).dot(theta[i])*(1-alpha)
             score[i]*=-0.5
             score[i]+=(theta[i].T).dot(h_test[i])
             score[i]-=0.5 
                
        
    elif model in ["kliep"]: 
        for i in range(n_nodes):
            phi_test.append(kernel[i].k_V(data_test[i]))
            phi_ref.append(kernel[i].k_V(data_ref[i]))
            
        for i in range(n_nodes):
            score[i]=np.mean(np.log(phi_test[i].dot(theta[i])+1e-6))
            

    elif model in ["grulsif","pool"]:
        for i in range(n_nodes):
            phi_test.append(kernel.k_V(data_test[i]))
            phi_ref.append(kernel.k_V(data_ref[i]))
        
        N_ref=len(phi_ref[0])
        N_test=len(phi_test[0])
        h_test =np.zeros((n_nodes,kernel.n))
        H_ref = np.zeros((n_nodes,kernel.n,kernel.n)) 
        H_test=np.zeros((n_nodes,kernel.n,kernel.n))
        
        for i in range(n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H_ref[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H_test[i]=np.einsum('ji,j...',  phi_test[i],phi_test[i])
            H_ref[i]/=N_ref 
            H_test[i]/=N_test
        score=np.zeros(n_nodes)
        
        for i in range(n_nodes):
            score[i]=theta[i].dot(H_test[i]).dot(theta[i])*alpha
            score[i]+=theta[i].dot(H_ref[i]).dot(theta[i])*(1-alpha)
            score[i]*=-0.5
            score[i]+=theta[i].dot(h_test[i])
            score[i]-=0.5 
  
    return score    
            
        
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        