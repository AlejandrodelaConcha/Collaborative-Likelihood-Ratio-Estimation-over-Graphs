# -----------------------------------------------------------------------------------------------------------------
# Title:  two_sample_test_univariate_models.py
# Author(s): Alejandro de la Concha   
# Initial version:  2020-05-17
# Last modified:    2025-02-27              
# This version:     2025-02-27
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to implement the two-sample tests associated with ULSIF, RULSIF, KLIEP, and MMD.
# -----------------------------------------------------------------------------------------------------------------
# Comments: This implementation is an adaptation of the code available at  
#           http://www.ms.k.u-tokyo.ac.jp/sugi/software.ht.
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy, Models, scipy, joblib, dill, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: RULSIF, ULSIF, KLIEP, MMD, f-divergence, hypothesis testing
# -----------------------------------------------------------------------------------------------------------------


from Models.aux_functions import *
from Models.likelihood_ratio_univariate import *
from Models.MMD_methods import *
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import dill as pickle
from pygsp import graphs,filters,plotting
from itertools import product
from scipy.sparse import csr_matrix


def run_permutation_MMD(permutation,data,N_ref,N_test,sigmas):
    ###### This function runs MMD for a given permutation.

    ### Input
    # permutation: the permutation of the indices used to run the method.
    # data: the concatenated data of size N_ref + N_test.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.
    # sigmas: sigma parameters for MMD.
    
    ### Output
    # scores: the MMD value associated with each of the nodes.

    
    n_nodes=len(data)
    scores=np.zeros(n_nodes)
    data_ref=[d[permutation][:N_ref] for d in data]
    data_test=[d[permutation][N_ref:] for d in data]
    scores=np.zeros(n_nodes)
    for i in range(n_nodes):
        aux_data=np.vstack((data_ref[i],data_test[i]))
        K=product_gaussian(sigmas[i],aux_data,aux_data)
        scores[i]=MMD(K,N_ref,N_test)  
    return scores
    
def run_permutation_RULSIF(permutation,phi,alpha,gammas,N_ref,N_test):
    ###### This function runs RULSIF for a given permutation.

    ### Input
    # permutation: the permutation of the indices used to run the method.
    # phi: precomputed matrix of feature maps to reduce computational cost.
    # alpha: regularization parameter related to the likelihood value.
    # gammas: regularization constants associated with the norm in H.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.

    ### Output
    # scores: the Pearson divergence associated with each of the nodes.

    
    n_nodes=len(phi)
    scores=np.zeros(n_nodes)

    for i in range(n_nodes):
        h = np.sum(phi[i][permutation][N_ref:], axis=0)
        h/= N_test
        H=np.einsum('ji,j...',phi[i][permutation][:N_ref] ,phi[i][permutation][:N_ref])*((1-alpha)/N_ref)
        H+=np.einsum('ji,j...',phi[i][permutation][N_ref:],phi[i][permutation][N_ref:])*(alpha/N_test) 
        n_centers=len(h)
        theta = np.linalg.solve(H+gammas[i]*np.eye(n_centers), h)
        scores[i]=theta.dot(H).dot(theta)
        scores[i]*=-0.5
        scores[i]+=theta.dot(h)
        scores[i]-=0.5 

    return scores

def run_permutation_ULSIF(permutation,phi,gammas,N_ref,N_test): 
    ###### This function runs ULSIF for a given permutation.

    ### Input
    # permutation: the permutation of the indices used to run the method.
    # phi: precomputed matrix of feature maps to reduce computational cost.
    # gammas: the regularization constant associated with the norm in H.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.

    ### Output
    # scores: the Pearson divergence associated with each of the nodes.
  
    n_nodes=len(phi)
    scores=np.zeros(n_nodes)

    for i in range(n_nodes):
        h = np.sum(phi[i][permutation][N_ref:], axis=0)
        h/= N_test
        H=np.einsum('ji,j...',phi[i][permutation][:N_ref] ,phi[i][permutation][:N_ref])/N_ref
        n_centers=len(h)
        theta = np.linalg.solve(H+gammas[i]*np.eye(n_centers), h)
        scores[i]=theta.dot(H).dot(theta)
        scores[i]*=-0.5
        scores[i]+=theta.dot(h)
        scores[i]-=0.5 

    return scores

def run_permutation_KLIEP(permutation,phi,tol,lr,N_ref,N_test): 
    ###### This function computes the Kullback–Leibler divergence for a given permutation.

    ### Input
    # permutation: the permutation of the indices used to run the method.
    # phi: precomputed matrix of feature maps to reduce computational cost.
    # tol: the tolerated level of error.
    # lr: the learning rate of the method.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.

    ### Output
    # score: the Kullback–Leibler divergence associated with each of the nodes.
    
    n_nodes=len(phi)
    scores=np.zeros(n_nodes)

    for i in range(n_nodes):
        theta=fit_kliep(phi[i][permutation][:N_ref],phi[i][permutation][N_ref:],tol,lr)
        scores[i]=np.mean(np.log(phi[i][permutation][N_ref:].dot(theta)))
        
    return scores
  
def fit_kliep(phi_ref,phi_test,tol=1e-3,lr=1e-3,verbose=False):
    ###### This auxiliary function fits KLIEP at a given tolerance rate.

    ### Input
    # phi_ref: the feature map associated with observations coming from p_v.
    # phi_test: the feature map associated with observations coming from q_v.
    # tol: the tolerated level of error.
    # lr: the learning rate of the method.
    # verbose: whether or not to display the optimization process.

    ### Output
    # theta: the parameter associated with the likelihood ratio.

    
    EPS=1e-6 ### variable to avoid overload errors 
    n_centers=len(phi_ref[0])
    b= np.mean(phi_ref, axis=0)
    b = b.reshape(-1, 1)
    theta= np.ones((n_centers, 1)) / n_centers
    previous_objective = -np.inf
    objective = np.mean(np.log(np.dot(phi_test, theta) + EPS))
    if verbose:
            print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
    k = 0
    while objective-previous_objective > tol:
        previous_objective = objective
        theta_p = np.copy(theta)
        theta += lr * np.dot(
            np.transpose(phi_test), 1./(np.dot(phi_test, theta) + EPS)
        )
        theta += b * ((((1-np.dot(np.transpose(b), theta)) /
                        (np.dot(np.transpose(b), b) + EPS))))
        theta = np.maximum(0, theta)
        theta /= (np.dot(np.transpose(b), theta) + EPS)
        objective = np.mean(np.log(np.dot(phi_test, theta) + EPS))
        k += 1
        if verbose:
            if k%100 == 0:
                print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))
                
    return theta
                
        
           
class RULSIF_two_sample_test():
######### This class implements the two-sample test at the node level via a permutation test based on the RULSIF algorithm.
    def __init__(self,data_ref,data_test,alpha,verbose=False,time=False):
        ### Input
        # data_ref: data points representing the distribution p_v(.).
        # data_test: data points representing the distribution q_v(.).
        # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
        # verbose: whether or not to print intermediate steps.
        # time: whether or not the time component is considered in the test.
        
        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
    
        try:
            alpha = float(alpha)
            if not (0.0 <= alpha < 1):
                raise ValueError(f"Parameter alpha must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: alpha parameter should be a float")
            sys.exit(1)            

        self.time=time 
        
        if time: 
            self.n_times=data_ref[0].shape[0]
            self.N_ref=data_ref[0].shape[1]
            self.N_test=data_test[0].shape[1]
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(data_ref[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]
            data_test=[transform_data(data_test[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]       
            self.n_nodes=len(data_ref)
             
        else:
            self.N_ref=len(data_ref[0])
            self.N_test=len(data_test[0])
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(d) for d in data_ref]
            data_test=[transform_data(d) for d in data_test]
            
        self.alpha=alpha
        
        RULSIF_1_models=[]
        RULSIF_2_models=[]
        self.score_pq=np.zeros(self.n_nodes)
        self.score_qp=np.zeros(self.n_nodes)
        self.gammas_1=np.zeros(self.n_nodes)
        self.gammas_2=np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            if verbose:
                print("Fitting RULSIF model for node %i"%i)
            RULSIF_1_models.append(RULSIF(data_ref[i],data_test[i],alpha=self.alpha))
            RULSIF_2_models.append(RULSIF(data_test[i],data_ref[i],alpha=self.alpha))
            self.score_pq[i]=RULSIF_1_models[i].PE_divergence(data_ref[i],data_test[i])
            self.score_qp[i]=RULSIF_2_models[i].PE_divergence(data_test[i],data_ref[i])
            self.gammas_1[i]=RULSIF_1_models[i].gamma
            self.gammas_2[i]=RULSIF_2_models[i].gamma
                       
        phi_ref_1=[RULSIF_1_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_1=[RULSIF_1_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]
        phi_ref_2=[RULSIF_2_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_2=[RULSIF_2_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]

        self.concatenate_data_1=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_1,phi_test_1)]
        self.concatenate_data_2=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_2,phi_test_2)]
        
    def fit(self,phi_ref,phi_test,gammas):
        ##### This function returns the theta parameters related to the likelihood ratios 
        ##### and the Pearson divergence at each node, computed by integrating the feature maps phi_ref and phi_test.
        ### Input
        # phi_ref: the feature map over the set X.
        # phi_test: the feature map over the set X'.
        # gammas: the regularization constants associated with the Hilbert norm.
        ### Output
        # theta: a numpy matrix of dimension n_nodes x L, where L is the size of the dictionary. 
        #        Each row represents a node in the graph.
        # score: a vector of dimension n_nodes, where each entry is the Pearson divergence estimate associated with that node.

        
        n_nodes = len(phi_ref)
        n_centers= phi_ref[0].shape[1]    
        scores=np.zeros(n_nodes)
        thetas=[]

        for i in range(n_nodes):
            h = np.sum(phi_test[i], axis=0)
            h/= self.N_test
            H=np.einsum('ji,j...',phi_ref[i],phi_ref[i])*((1-self.alpha)/self.N_ref)
            H+=np.einsum('ji,j...',phi_test[i],phi_test[i])*(self.alpha/self.N_test) 
            thetas.append(np.linalg.solve(H+gammas[i]*np.eye(n_centers), h))
            scores[i]=thetas[i].dot(H).dot(thetas[i])
            scores[i]*=-0.5
            scores[i]+=thetas[i].dot(h)
            scores[i]-=0.5 
             
        return thetas,scores

    def aux_get_divergences(self,permutations):
        ### This function computes the Pearson divergence associated with a set of permutations.
        ### Input:
        # permutations: a list of permutations.
        ### Output:
        # PE_1, PE_2: lists of Pearson divergence scores, depending on the order in which the datasets are taken.
        
        n_permutations=permutations.shape[0]
        n_nodes=len(self.concatenate_data_1)
        PE_1=np.zeros((n_permutations,n_nodes))
        PE_2=np.zeros((n_permutations,n_nodes))
        
        for i in range(n_permutations):
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_1]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_1]
            _,PE_1[i]= self.fit(hat_phi_ref, hat_phi_test,self.gammas_1)
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_2]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_2]
            _,PE_2[i]= self.fit(hat_phi_test, hat_phi_ref,self.gammas_2)
            
        return PE_1,PE_2
    
    def run_permutations(self,n_permutations):
        #### Function generating n_permutations over the index set {0, 1, ..., N_ref + N_test}.
        ### Input:
        # n_permutations: the number of permutations to generate.
        ### Output:
        # permutations: a list of generated permutations.

        
        permutations=np.zeros((n_permutations,self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i]=np.random.permutation(self.N_ref+self.N_test)             
        permutations=permutations.astype(int)
        
        return permutations    
    
    def get_pivalues(self,n_rounds=1000):
        #### The p-values are computed via a permutation test.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.
    
        n_nodes=len(self.score_pq)

        permutations=self.run_permutations(n_rounds)
        PD_1,PD_2=self.aux_get_divergences(permutations)

        self.PD_1s=PD_1
        self.PD_2s=PD_2
        
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)

        for i in range(n_rounds):
            scores_pq[i] = np.max(self.PD_1s[i])
            scores_qp[i] = np.max(self.PD_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2

    
    def get_pivalues_multiprocessing(self,n_rounds=1000):
        #### The p-values are computed via a permutation test. When the equipment allows it, multiprocessing is used to speed up computations.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.
   
        permutations=self.run_permutations(n_rounds)

        PD_1=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_RULSIF)(p,self.concatenate_data_1,self.alpha,
                                                                                  self.gammas_1,self.N_ref,self.N_test) for p in permutations)
        
        PD_2=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_RULSIF)(p,self.concatenate_data_2,self.alpha,
                                                                                  self.gammas_2,self.N_ref,self.N_test) for p in permutations)
        
        self.PD_1s=PD_1
        self.PD_2s=PD_2
        
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)

        for i in range(n_rounds):
            scores_pq[i] = np.max(self.PD_1s[i])
            scores_qp[i] = np.max(self.PD_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2

         

    
class LSTT():
######### This class implements the two sampling test at the node level via a permutation test based on the ULSIF algorithm
    def __init__(self,data_ref,data_test,verbose=False,time=False):
        ### Input
        # data_ref: data points representing the distribution p_v(.)
        # data_test: data points representing the distribution q_v(.)
        # alpha: regularization parameter associated with the upperbound of the likelihood ratio
        # verbose: whether or not to print intermediate steps.
        # time: whether the time component is considered in the test or not

        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
    
        
        self.time=time 
        
        if time: 
            self.n_times=data_ref[0].shape[0]
            self.N_ref=data_ref[0].shape[1]
            self.N_test=data_test[0].shape[1]
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(data_ref[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]
            data_test=[transform_data(data_test[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]       
            self.n_nodes=len(data_ref)
             
        else:
            self.N_ref=len(data_ref[0])
            self.N_test=len(data_test[0])
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(d) for d in data_ref]
            data_test=[transform_data(d) for d in data_test]

        
        ULSIF_1_models=[]
        ULSIF_2_models=[]
        self.score_pq=np.zeros(self.n_nodes)
        self.score_qp=np.zeros(self.n_nodes)
        self.gammas_1=np.zeros(self.n_nodes)
        self.gammas_2=np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            if verbose:
                print("Fitting ULSIF model for node %i"%i)
            ULSIF_1_models.append(ULSIF(data_ref[i],data_test[i]))
            ULSIF_2_models.append(ULSIF(data_test[i],data_ref[i]))
            self.score_pq[i]=ULSIF_1_models[i].PE_divergence(data_ref[i],data_test[i])
            self.score_qp[i]=ULSIF_2_models[i].PE_divergence(data_test[i],data_ref[i])
            self.gammas_1[i]=ULSIF_1_models[i].gamma
            self.gammas_2[i]=ULSIF_2_models[i].gamma
                       
        phi_ref_1=[ULSIF_1_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_1=[ULSIF_1_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]
        phi_ref_2=[ULSIF_2_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_2=[ULSIF_2_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]

        self.concatenate_data_1=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_1,phi_test_1)]
        self.concatenate_data_2=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_2,phi_test_2)]
        
    def fit(self,phi_ref,phi_test,gammas):
        ##### This functions returns the theta parameters related to the likelihood ratios and the PEARSON divergence at each node
        ##### comupted by integrating the feature maps phi_ref,phi_test 
        ### Input
        # phi_ref: the feature map over the set X
        # phi_test: the feature map over the set X'
        # gammas: the penalization constants associated to the Hilbert norm
        ### Output
        # theta: a numpy matrix of dimension n_nodesxL, where L is the size of the dictionary. Each row represents a node in the graph
        # score: vector of dimension n_nodes, each entry is PEARSON divergence estimate associated to that node
                
        n_nodes = len(phi_ref)
        n_centers= phi_ref[0].shape[1]          
        scores=np.zeros(n_nodes)
        thetas=[]

        for i in range(n_nodes):
            h = np.sum(phi_test[i], axis=0)
            h/= self.N_test
            H=np.einsum('ji,j...',phi_ref[i],phi_ref[i])/self.N_ref
            thetas.append(np.linalg.solve(H+gammas[i]*np.eye(n_centers), h))
            scores[i]=thetas[i].dot(H).dot(thetas[i])
            scores[i]*=-0.5
            scores[i]+=thetas[i].dot(h)
            scores[i]-=0.5 
             
        return thetas,scores

    def aux_get_divergences(self,permutations):
        ### This function obtains the PEARSON divergence associated to a set of permutations.
        
        ### Input:
        # permutations: a list of permutations
        ### Output
        # PE_1,PE_2: a list of PEARSON divergence scores depending on the order the datasets are taken 

        n_permutations=permutations.shape[0]
        n_nodes=len(self.concatenate_data_1)
        PE_1=np.zeros((n_permutations,n_nodes))
        PE_2=np.zeros((n_permutations,n_nodes))
        
        for i in range(n_permutations):
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_1]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_1]
            _,PE_1[i]= self.fit(hat_phi_ref, hat_phi_test,self.gammas_1)
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_2]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_2]
            _,PE_2[i]= self.fit(hat_phi_test, hat_phi_ref,self.gammas_2)
            
        return PE_1,PE_2
    
    def run_permutations(self,n_permutations):
        #### Function generating n_permutations over the index set 0,1,....,N_ref+N_test
        ### Input
        # n_permutations: number of permutations to generate
        ### Output
        # permutations: list of generated permutations 
                
        permutations=np.zeros((n_permutations,self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i]=np.random.permutation(self.N_ref+self.N_test)             
        permutations=permutations.astype(int)
        
        return permutations    
    
    def get_pivalues(self,n_rounds=1000):
        #### The p-values are computed via permutation test.
        # Input
        # n_rounds: Number of permutations used to estimated the p-values
        # Output
        # p-values_1,p_values_2: p_values associated to each side of the PEARSON divergence estimates.
             
        n_nodes=len(self.score_pq)  
        permutations=self.run_permutations(n_rounds)
        PD_1,PD_2=self.aux_get_divergences(permutations)

        self.PD_1s=PD_1
        self.PD_2s=PD_2
         
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)
        for i in range(n_rounds):
            scores_pq[i] = np.max(self.PD_1s[i])
            scores_qp[i] = np.max(self.PD_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2


    
    def get_pivalues_multiprocessing(self,n_rounds=1000):
        #### The p-values are computed via permutation test. When the equipment allow it multiprocessing is used to speed computations.
        # Input
        # n_rounds: Number of permutations used to estimated the p-values
        # Output
        # p-values_1,p_values_2: p_values associated to each side of the PEARSON divergence estimates.
           
        permutations=self.run_permutations(n_rounds)
        PD_1=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_ULSIF)(p,self.concatenate_data_1,
                                                                                  self.gammas_1,self.N_ref,self.N_test) for p in permutations)       
        PD_2=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_ULSIF)(p,self.concatenate_data_2,
                                                                                  self.gammas_2,self.N_ref,self.N_test) for p in permutations)
        self.PD_1s=PD_1
        self.PD_2s=PD_2
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)
        for i in range(n_rounds):
            scores_pq[i] = np.max(self.PD_1s[i])
            scores_qp[i] = np.max(self.PD_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2




class KLIEP_two_sample_test():
######### This class implements the two sampling test at the node level via a permutation test based on the KLIEP algorithm    
    def __init__(self,data_ref,data_test,tol=1e-2,lr=1e-4,verbose=False,time=False):
        ### Input
        # data_ref: data points representing the distribution p_v(.).
        # data_test: data points representing the distribution q_v(.).
        # tol: the level of tolerated error.
        # lr: the learning rate of the method.
        # verbose: whether or not to print intermediate steps.
        # time: whether or not the time component is considered in the test.
        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
    
        
        self.time=time 
        self.tol=tol
        self.lr=lr
        
        if time: 
            self.n_times=data_ref[0].shape[0]
            self.N_ref=data_ref[0].shape[1]
            self.N_test=data_test[0].shape[1]
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(data_ref[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]
            data_test=[transform_data(data_test[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]       
            self.n_nodes=len(data_ref)
             
        else:
            self.N_ref=len(data_ref[0])
            self.N_test=len(data_test[0])
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(d) for d in data_ref]
            data_test=[transform_data(d) for d in data_test]
        
        
        KLIEP_1_models=[]
        KLIEP_2_models=[]
        self.score_pq=np.zeros(self.n_nodes)
        self.score_qp=np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            KLIEP_1_models.append(KLIEP(data_ref[i],data_test[i],lr=self.lr,tol=self.tol))
            KLIEP_2_models.append(KLIEP(data_test[i],data_ref[i],lr=self.lr,tol=self.tol))
            self.score_pq[i]=KLIEP_1_models[i].KL_divergence(data_ref[i],data_test[i])
            self.score_qp[i]=KLIEP_2_models[i].KL_divergence(data_test[i],data_ref[i])
            
        phi_ref_1=[KLIEP_1_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_1=[KLIEP_1_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]
        phi_ref_2=[KLIEP_2_models[i].kernel.k_V(data_ref[i]) for i in range(len(data_ref))]
        phi_test_2=[KLIEP_2_models[i].kernel.k_V(data_test[i]) for i in range(len(data_test))]  
        
        self.concatenate_data_1=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_1,phi_test_1)]
        self.concatenate_data_2=[np.vstack((d_r,d_t)) for d_r,d_t in zip(phi_ref_2,phi_test_2)]

        
    def fit(self,phi_ref,phi_test,verbose=False):
        ##### This function returns the theta parameters related to the likelihood ratios 
        ##### and the Kullback-Leibler divergence at each node, computed by integrating the feature maps phi_ref and phi_test.
        ### Input
        # phi_ref: the feature map over the set X.
        # phi_test: the feature map over the set X'.
        # verbose: whether or not to plot the model selection results.
        ### Output
        # theta: a numpy matrix of dimension n_nodes x L, where L is the size of the dictionary. 
        #        Each row represents a node in the graph.
        # score: a vector of dimension n_nodes, where each entry is the Kullback-Leibler divergence estimate associated with that node.

        
        EPS=1e-6 ### variable to avoid overload errors              
        n_nodes=len(phi_ref) 
        n_centers= phi_ref[0].shape[1]     
        scores=np.zeros(n_nodes)
        thetas=[]
        for i in range(n_nodes):   

            b= np.mean(phi_ref[i], axis=0)
            b = b.reshape(-1, 1)
            theta= np.ones((n_centers, 1)) / n_centers
            previous_objective = -np.inf
            objective = np.mean(np.log(np.dot(phi_test[i], theta) + EPS))
            if verbose:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
            k = 0
            while objective-previous_objective > self.tol:
            
                previous_objective = objective
                theta_p = np.copy(theta)
                theta += self.lr * np.dot(
                    np.transpose(phi_test[i]), 1./(np.dot(phi_test[i], theta) + EPS)
                    )
                theta += b * ((((1-np.dot(np.transpose(b), theta)) /
                            (np.dot(np.transpose(b), b) + EPS))))
                theta[theta<0] = 0
                theta /= (np.dot(np.transpose(b), theta) + EPS)
                objective = np.mean(np.log(np.dot(phi_test[i], theta) + EPS))
                k += 1
                if verbose:
                    if k%100 == 0:
                        print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))
                        
            thetas.append(1.0*theta)
            scores[i]=np.mean(np.log(phi_test[i].dot(theta)+ EPS))
            
        return thetas,scores
       
    
        
    def aux_get_divergences(self,permutations):
        ### This function computes the Kullback-Leibler divergence associated with a set of permutations.
        ### Input:
        # permutations: a list of permutations.
        ### Output:
        # KL_1, KL_2: lists of Kullback-Leibler divergence scores, depending on the order in which the datasets are taken.


        n_permutations=permutations.shape[0]
        n_nodes=len(self.concatenate_data_1)
        KL_1=np.zeros((n_permutations,n_nodes))
        KL_2=np.zeros((n_permutations,n_nodes))
        
        for i in range(n_permutations):
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_1]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_1]
            _,KL_1[i]= self.fit(hat_phi_ref, hat_phi_test)
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data_2]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data_2]
            _,KL_2[i]= self.fit(hat_phi_test, hat_phi_ref)
            
        return KL_1,KL_2
    
    
    def run_permutations(self,n_permutations):
        #### Function generating n_permutations over the index set {0, 1, ..., N_ref + N_test}.
        ### Input:
        # n_permutations: the number of permutations to generate.
        ### Output:
        # permutations: a list of generated permutations.

        permutations=np.zeros((n_permutations,self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i]=np.random.permutation(self.N_ref+self.N_test)             
        permutations=permutations.astype(int)
        
        return permutations    
    
    def get_pivalues(self,n_rounds=1000):
        #### The p-values are computed via a permutation test.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Kullback-Leibler divergence estimates.
        
        n_nodes=len(self.score_pq)

        permutations=self.run_permutations(n_rounds)
        KL_1,KL_2=self.aux_get_divergences(permutations)

        self.KL_1s=KL_1
        self.KL_2s=KL_2
        
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)
        for i in range(n_rounds):
            scores_pq[i] = np.max(self.KL_1s[i])
            scores_qp[i] = np.max(self.KL_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2
                  
        
    
    def get_pivalues_multiprocessing(self,n_rounds=1000):
        #### The p-values are computed via a permutation test. When the equipment allows it, multiprocessing is used to speed up computations.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.

           
        permutations=self.run_permutations(n_rounds)

        KL_1=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_KLIEP)(permutation=p,phi=self.concatenate_data_1,
                                                                                  N_ref=self.N_ref,N_test=self.N_test,
                                                                                  tol=self.tol,lr=self.lr) for p in permutations)     
        KL_2=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_KLIEP)(permutation=p,phi=self.concatenate_data_2,
                                                                                  N_ref=self.N_ref,N_test=self.N_test,
                                                                                  tol=self.tol,lr=self.lr) for p in permutations)
             
        self.KL_1s=KL_1
        self.KL_2s=KL_2
        
        n_nodes = len(self.score_pq)
        scores_pq = np.zeros(n_rounds)
        scores_qp = np.zeros(n_rounds)
        for i in range(n_rounds):
            scores_pq[i] = np.max(self.KL_1s[i])
            scores_qp[i] = np.max(self.KL_2s[i])

        p_values_1 = np.zeros(n_nodes)
        p_values_2 = np.zeros(n_nodes)

        for i in range(n_nodes):
            p_values_1[i] = np.mean(scores_pq>=self.score_pq[i])
            p_values_2[i] = np.mean(scores_qp>=self.score_qp[i])
            
        return p_values_1,p_values_2
                  

     
class MMD_two_sample_test():
### This class implements the two-sample test at the node level via a permutation test based on the MMD algorithm.
    def __init__(self,data_ref,data_test,estimate_sigma=False,verbose=False,time=False):
        ### Input
        # data_ref: data points representing the distribution p_v(.).
        # data_test: data points representing the distribution q_v(.).
        # estimate_sigma: whether the sigma parameter will be computed to optimize the power of the test. 
        #                 If False, the median heuristic is used.
        # time: whether or not there is a temporal aspect to consider.
        # verbose: whether or not to print intermediate steps.
 

        self.time=time 
        
        if time: 
            self.n_times=data_ref[0].shape[0]
            self.N_ref=data_ref[0].shape[1]
            self.N_test=data_test[0].shape[1]
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(data_ref[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]
            data_test=[transform_data(data_test[node][t]) for t,node in list(product(range(self.n_times),range(self.n_nodes)))]       

             
        else:
            self.N_ref=len(data_ref[0])
            self.N_test=len(data_test[0])
            self.n_nodes=len(data_ref)
            data_ref=[transform_data(d) for d in data_ref]
            data_test=[transform_data(d) for d in data_test]
            
        self.mmd_models=MMD_nodes(data_ref,data_test,estimate_sigma)
            
        self.scores=self.mmd_models.get_MMD()

        self.concatenate_data=[np.vstack((d_r,d_t)) for d_r,d_t in zip(data_ref,data_test)]
         
    def fit(self,data_ref,data_test): 
        ### It computes the the MMD score associated to each of the nodes. 
        # data_ref: data points representing the distribution p_v(.)
        # data_test: data points representing the distribution q_v(.)        
        
        return self.mmd_models.get_MMD(data_ref,data_test)

    def aux_get_divergences(self,permutations):
        ### This function obtains the MMD associated to a set of permutations.
        ### Input:
        # permutations: a list of permutations
        ### Output
        # MMDs: a list of MMD scores associated to a list of permutations

        n_permutations=permutations.shape[0]
        n_nodes=len(self.concatenate_data)
        MMD_s=np.zeros((n_permutations,n_nodes))
   
        for i in range(n_permutations):
            hat_phi_ref= [d[permutations[i]][:self.N_ref] for d in self.concatenate_data]
            hat_phi_test=[d[permutations[i]][self.N_ref:] for d in self.concatenate_data]
            MMD_s[i]= self.fit(hat_phi_ref, hat_phi_test)
      
        return MMD_s
    
    def run_permutations(self,n_permutations):
        #### Function generating n_permutations over the index set {0, 1, ..., N_ref + N_test}.
        ### Input:
        # n_permutations: the number of permutations to generate.
        ### Output:
        # permutations: a list of generated permutations.

        
        permutations=np.zeros((n_permutations,self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i]=np.random.permutation(self.N_ref+self.N_test)             
        permutations=permutations.astype(int)
        
        return permutations    
    
    def get_pivalues(self,n_rounds=1000):
        #### The p-values are computed via a permutation test.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values: p-values associated with the MMD estimates.

          
        permutations=self.run_permutations(n_rounds)
        self.MMD_s=self.aux_get_divergences(permutations)
    
        scores= np.zeros(n_rounds)
        for i in range(n_rounds):
            scores[i] = np.max(self.MMD_s[i])
        p_values = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            p_values[i] = np.mean(scores>=self.scores[i])

        return p_values
    
    def get_pivalues_multiprocessing(self,n_rounds=1000):
        #### The p-values are computed via a permutation test. When the equipment allows it, multiprocessing is used to speed up computations.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values: p-values associated with the MMD estimates.

        
        permutations=self.run_permutations(n_rounds)
                    
        MMD_s=Parallel(n_jobs=-1,prefer="threads")(delayed(run_permutation_MMD)(p,self.concatenate_data,self.N_ref,self.N_test,self.mmd_models.sigmas) for p in permutations)
        self.MMD_s=MMD_s
     
        scores= np.zeros(n_rounds)
        for i in range(n_rounds):
            scores[i] = np.max(self.MMD_s[i])
        p_values = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            p_values[i] = np.mean(scores>=self.scores[i])

        return p_values


    
    
    
    
    
    
   