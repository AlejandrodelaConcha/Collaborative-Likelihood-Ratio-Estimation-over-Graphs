# -----------------------------------------------------------------------------------------------------------------
# Title:  Experiments_two_sample_test
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-27              
# This version:     2025-02-27
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to generate the experiments presented in the paper 
#               "Collaborative Two-Sample Testing."
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, scipy, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: collaborative two-sample testing, C2ST
# -----------------------------------------------------------------------------------------------------------------


import numpy as np
from scipy.stats import multivariate_normal,norm
from pygsp import graphs, filters, plotting
import itertools
import pickle


def generate_experiment_1(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=False):
    
    ######### This function implements the first experiment of the paper.

    ### Input:
    # N_ref: Number of observations from p_v.
    # N_test: Number of observations from p'_v.
    # p_within: Probability of connection between nodes within the same cluster.
    # p_between: Probability of connection between nodes from different clusters.
    # seed: The seed used to generate the stochastic block model.
    # H_null: Boolean indicating whether there are some nodes where the null hypothesis is satisfied.

    ### Output:
    # G: The graph component of the problem.
    # data_ref: A list of n_nodes elements, each representing the observations coming from p_v.
    # data_test: A list of n_nodes elements, each representing the observations coming from q_v.
    # affected_nodes: A NumPy array of the nodes affected during the experiment.

    
    n_nodes=100
    n_clusters=4
    z=[[i]*int(n_nodes/n_clusters) for i in range(0,n_clusters)]
    z=list(itertools.chain.from_iterable(z))
    G=graphs.StochasticBlockModel(N=n_nodes,k=n_clusters,z=z,q=p_between,seed=seed,p=p_within) 
    
    data_ref=[np.random.normal(size=N_ref) for i in range(n_nodes)]
    data_test=[np.random.normal(size=N_test) for i in range(n_nodes)]
    
    if H_null:
        affected_nodes=[]
        
    else:

        for i in np.arange(25):
            data_test[i]=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=N_test)
    
        for i in np.arange(75,100):
            data_test[i]=np.random.normal(loc=1,size=N_test)
    
        affected_nodes=np.hstack((np.arange(25), np.arange(75,100)))
        
        
    return G,data_ref,data_test,affected_nodes


def generate_experiment_2(N_ref,N_test,p_within=0.5,p_between=0.01,seed=0,H_null=False):
    
    ######### This function implements the second experiment of the paper.
    ### Input:
    # N_ref: Number of observations from p_v.
    # N_test: Number of observations from p'_v.
    # p_within: Probability of connection between nodes within the same cluster.
    # p_between: Probability of connection between nodes from different clusters.
    # seed: The seed used to generate the stochastic block model.
    # H_null: Boolean indicating whether there are some nodes where the null hypothesis is satisfied.
    ### Output:
    # G: The graph component of the problem.
    # data_ref: A list of n_nodes elements, each representing the observations coming from p_v.
    # data_test: A list of n_nodes elements, each representing the observations coming from q_v.
    # affected_nodes: A NumPy array of the nodes affected during the experiment.

   
    n_nodes=100
    n_clusters=4
    sample_size=int(n_nodes/n_clusters)
    z=[[i]*int(n_nodes/n_clusters) for i in range(0,n_clusters)]
    z=list(itertools.chain.from_iterable(z))
    G=graphs.StochasticBlockModel(N=n_nodes,k=n_clusters,z=z,q=p_between,seed=seed,p=p_within) 
    
    mean=np.zeros(2)
    sigma_1_1=np.array([1,-4/5,-4/5,1]).reshape((2,2))
    sigma_2_1=np.array([1,4/5,4/5,1]).reshape((2,2))

    sigma_1_2=np.array([1,-4/5,-4/5,1]).reshape((2,2))
    sigma_2_2=np.array([1,0,0,1]).reshape((2,2))

    sigma_1_3=np.array([1,4/5,4/5,1]).reshape((2,2))
    sigma_2_3=np.array([1,0,0,1]).reshape((2,2))

    mu_4=1*np.ones(2)
    sigma_1_4=np.array([1,0,0,1]).reshape((2,2))
    
    data_ref=[]
    data_test=[]
    
    for i in range(25):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_1,size=N_ref)) 
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_1,size=N_test))  
        
    for i in range(25,50):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_2,size=N_ref)) 
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_2,size=N_test))
        
    for i in range(50,75):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_3,size=N_ref))
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_3,size=N_test))
        
    for i in range(75,100):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_4,size=N_ref))
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_4,size=N_test))
   
    if H_null:
        affected_nodes=[] 
        
    else:
        for i in range(2*sample_size,3*sample_size):
            data_test[i]=multivariate_normal.rvs(mean=mean,cov=sigma_2_3,size=N_test)
            
        for i in range(3*sample_size,4*sample_size):
            data_test[i]=multivariate_normal.rvs(mean=mu_4,cov=sigma_1_4,size=N_test)
       
        affected_nodes=np.hstack((np.arange(2*sample_size,3*sample_size),
                                  np.arange(3*sample_size,4*sample_size)))
     
    return G,data_ref,data_test,affected_nodes
 

def generate_experiment_3(N_ref,N_test,seed=0,H_null=False):
    
    ######### This function implements the third experiment of the paper.
    ### Input:
    # n_nodes: Number of nodes.
    # N_ref: Number of observations from p_v.
    # N_test: Number of observations from p'_v.
    # m0: Probability of connection between nodes within the same cluster.
    # m: Probability of connection between nodes from different clusters.
    # seed: The seed used to generate the stochastic block model.
    ### Output:
    # G: The graph component of the problem.
    # data_ref: A list of n_nodes elements, each representing the observations coming from p_v.
    # data_test: A list of n_nodes elements, each representing the observations coming from q_v.
    # affected_nodes: A NumPy array of the nodes affected during the experiment.

  
    n_nodes=100
  #  G = graphs.BarabasiAlbert(N=n_nodes,m0=5,seed=seed,m=2)
    N_1=int(np.sqrt(n_nodes))
    N_2=int(np.ceil(n_nodes/N_1))
    G=graphs.Grid2d(N1=N_1,N2=N_2)
    degrees = np.array(np.sum(G.W,axis=0))[0]
    neighbours = [np.where((G.W[i, :] > 0).todense())[1] for i in range(n_nodes)] 
    
    if seed is not None:
        np.random.seed(seed=seed)
        selected_node=np.random.choice(n_nodes,p=degrees/np.sum(degrees),size=1)[0]
        index_nodes_2hop=np.where(G.W.dot(G.W)[selected_node].todense()[0]>0)[1]
        affected_nodes=np.hstack((index_nodes_2hop,neighbours[selected_node]))
        affected_nodes=np.hstack((affected_nodes,selected_node))
        affected_nodes=np.unique(affected_nodes)
    else:
        selected_node=np.random.choice(n_nodes,p=degrees/np.sum(degrees),size=1)[0]
        index_nodes_2hop=np.where(G.W.dot(G.W)[selected_node].todense()[0]>0)[1]
        affected_nodes=np.hstack((index_nodes_2hop,neighbours[selected_node]))
        affected_nodes=np.hstack((affected_nodes,selected_node))
        affected_nodes=np.unique(affected_nodes)        
    np.random.seed()
    mu=np.zeros(3)
    sigma=np.eye(3)
    sigma[1,2]=4/5
    sigma[2,1]=4/5
    data_ref=[multivariate_normal.rvs(mean=mu,cov=sigma,size=N_ref) for i in range(n_nodes)]
    data_test=[multivariate_normal.rvs(mean=mu,cov=sigma,size=N_test) for i in range(n_nodes)]
    
    if H_null:
        affected_nodes=[] 
    else:
        sigma[1,2]=-4/5
        sigma[2,1]=-4/5
        for i in affected_nodes:
            data_test[int(i)]=multivariate_normal.rvs(mean=mu,cov=sigma,size=N_test)

    return G,data_ref,data_test,affected_nodes

def generate_experiment_4(N_ref,N_test,seed=0,H_null=False):
    
    ######### This function implements the fourth experiment of the paper.
    ### Input:
    # n_nodes: Number of nodes.
    # N_ref: Number of observations from p_v.
    # N_test: Number of observations from p'_v.
    # seed: The seed used to generate the stochastic block model.
    # H_null: Boolean indicating whether there are some nodes where the null hypothesis is satisfied.
    ### Output:
    # G: The graph component of the problem.
    # data_ref: A list of n_nodes elements, each representing the observations coming from p_v.
    # data_test: A list of n_nodes elements, each representing the observations coming from q_v.
    # affected_nodes: A NumPy array of the nodes affected during the experiment.

 
    n_nodes=100
    n_nodes=100
  #  G = graphs.BarabasiAlbert(N=n_nodes,m0=5,seed=seed,m=2)
    N_1=int(np.sqrt(n_nodes))
    N_2=int(np.ceil(n_nodes/N_1))
    G=graphs.Grid2d(N1=N_1,N2=N_2)
  #  G = graphs.BarabasiAlbert(N=n_nodes,m0=5,seed=seed,m=2)
    degrees = np.array(np.sum(G.W,axis=0))[0]
    neighbours = [np.where((G.W[i, :] > 0).todense())[1] for i in range(n_nodes)] 
    
    if seed is not None:
        np.random.seed(seed=seed)
        selected_node=np.random.choice(n_nodes,p=degrees/np.sum(degrees),size=1)[0]
        index_nodes_2hop=np.where(G.W.dot(G.W)[selected_node].todense()[0]>0)[1]
        affected_nodes=np.hstack((index_nodes_2hop,neighbours[selected_node]))
        affected_nodes=np.hstack((affected_nodes,selected_node))
        affected_nodes=np.unique(affected_nodes)
    else:
        selected_node=np.random.choice(n_nodes,p=degrees/np.sum(degrees),size=1)[0]
        index_nodes_2hop=np.where(G.W.dot(G.W)[selected_node].todense()[0]>0)[1]
        affected_nodes=np.hstack((index_nodes_2hop,neighbours[selected_node]))
        affected_nodes=np.hstack((affected_nodes,selected_node))
        affected_nodes=np.unique(affected_nodes)
         
    np.random.seed()
    data_ref=[np.random.multivariate_normal(mean=np.array([0,0]),cov=10*np.eye(2),size=N_ref) for i in range(n_nodes)]
    data_test=[multivariate_normal.rvs(mean=np.array([0,0]),cov=10*np.eye(2),size=N_test) for i in range(n_nodes)]

    if H_null:
        affected_nodes=[] 
    
    else:
        means=[np.array([0,0]),np.array([0,5]),np.array([0,-5]),np.array([5,0]),np.array([-5,0])]
        p=(1/5)*np.ones(5)
        for i in affected_nodes:
            gaussian_data=[]
            u=np.random.choice(np.arange(5),p=p,size=N_test)
            counts=np.unique(u,return_counts=True)
            for j in range(len(counts[0])):
                gaussian_data.append(np.random.multivariate_normal(mean=means[j],cov=5*np.eye(2),size=counts[1][j]))
            data_test[i]=np.vstack(gaussian_data)
            index=np.random.choice(N_test,size=N_test,replace=False)
            data_test[i]=data_test[i][index]
        

    return G,data_ref,data_test,affected_nodes








