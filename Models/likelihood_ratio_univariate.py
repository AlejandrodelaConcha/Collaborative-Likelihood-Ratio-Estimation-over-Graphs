# -----------------------------------------------------------------------------------------------------------------
# Title:  likelihood_ratio_univariate
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-25              
# Current version:  2025-02-25
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): 
# This code implements the methods described in Sugiyama (2007), Sugiyama (2011), and Yamada (2011).
# The implementation is based on the package: 
# https://github.com/adapt-python/adapt/tree/master/adapt
# and the implementation made available by the authors: 
# https://riken-yamada.github.io/RuLSIF.html
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, numba, Models.aux_functions
# -----------------------------------------------------------------------------------------------------------------
# Keywords: ULSIF_implementation, RuLSIF_implementation, KLIEP_implementation
# -----------------------------------------------------------------------------------------------------------------


from Models.aux_functions import *
import numpy as np
from numba import njit,jit
import copy

     

class RULSIF():
########## Class implementing the RULSIF f-divergence estimator 
    def __init__(self,data_ref,data_test,alpha=0.1,verbose=False):     
    
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution p_v'(.) 
    # alpha: regularization parameter associated with the upperbound of the likelihood ratio
    # verbose: whether or not print intermediate results  
        
        self.data_ref=transform_data(data_ref)
        self.data_test=transform_data(data_test)
        self.alpha=alpha
        kernel_1=self.initializalize_kernel() 
        self.kernel,self.gamma=self.model_selection(kernel_1,verbose)

     
    def initializalize_kernel(self):

    ## Output
    # kernel_1: A initialized kernel with a given dictionary 

        if self.data_test.shape[0]>100:
            index=np.random.choice(len(self.data_test),replace=False,size=100)
            dictionary=1*self.data_test[index]
        else:
            dictionary=1*self.data_test
            
        kernel_1=Gauss_Kernel(dictionary,gamma=1.0)
                   
        return kernel_1
    
    def fit(self,data_ref=None,data_test=None,kernel=None,gamma=None): 
    #### Function estimating the theta parameter for a given set of observations comming from p_v(.) and q_v(.) 
    
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution q_v(.) 
    # kernel: kernel to be used in the method
    # gamma: penalization constant related with the sparsness of the parameters
    
    ## Output
    # theta: the parameter related with likeliood-ratio estimation 
    
        if  data_ref is not None:
            data_ref=transform_data(data_ref)
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=transform_data(data_test)
        else:
            data_test=self.data_test
    
        if kernel is None:
            kernel=self.kernel
        
        if gamma is None:
            gamma=self.gamma
    
        n_centers=kernel.n
        phi_test=kernel.k_V(data_test)
        phi_ref=kernel.k_V(data_ref)
   
        N_test=len(phi_test)
        N_ref=len(phi_ref)
        
        H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+gamma*np.eye(n_centers), h)

        return theta
    
    def model_selection(self,kernel,verbose=False):
        
        ###### Input
        # kernel: kernel to be used in the method
        # verbose: whether or not to print intermediate steps 
        
        ###### Output
        # kernel_,gamma: kernel initialized with the selected dictionary and the optimal value of gamma
        
        sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])
        
        max_ = -np.inf
        j_scores_ = {}
        best_params_={}
        N_test=self.data_test.shape[0]
        N_ref=self.data_ref.shape[0]
        N_min=np.min((N_ref,N_test))
        gamma_list = np.logspace(-5,1,4)
        
        if N_ref<N_test:
            index_data = np.random.choice(
                            N_test,
                            N_ref,
                            replace=False)
        elif N_test<N_ref:
            index_data = np.random.choice(
                            N_ref,
                            N_test,
                            replace=False)
            
        
        for s in range(len(sigma_list)):
            kernel_=Gauss_Kernel(dictionary=kernel.dictionary,gamma=sigma_list[s])

            
            if N_ref<N_test:
                phi_test = kernel_.k_V(self.data_test[index_data])
                phi_ref  = kernel_.k_V(self.data_ref)
                
            elif N_test<N_ref:
                phi_test = kernel_.k_V(self.data_test)
                phi_ref =  kernel_.k_V(self.data_ref[index_data])
            else:
                phi_test =  kernel_.k_V(self.data_test)
                phi_ref =   kernel_.k_V(self.data_ref)
            
        
            H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref 
         
            h = np.mean(phi_test, axis=0)
          
            h = h.reshape(-1, 1)
            for g in gamma_list:
                B = H + np.identity(kernel.n) * (g * (N_test - 1) / N_test)
                BinvX = np.linalg.solve(B, phi_test.T)
                XBinvX = phi_test.T * BinvX
                D0 = np.ones(N_min) * N_test- np.dot(np.ones(kernel.n), XBinvX)
                diag_D0 = np.diag((np.dot(h.T, BinvX) / D0).ravel())
                B0 = np.linalg.solve(B, h * np.ones(N_min)) + np.dot(BinvX, diag_D0)
                diag_D1 = np.diag(np.dot(np.ones(kernel.n), phi_ref.T * BinvX).ravel())
                B1 = np.linalg.solve(B,  phi_ref.T) + np.dot(BinvX, diag_D1)
                B2 = (N_test- 1) * (N_ref* B0 - B1) / (N_test* (N_ref - 1))
                B2[B2<0]=0
                r_s = (phi_ref.T * B2).sum(axis=0).T
                r_t= (phi_test.T * B2).sum(axis=0).T   
                score = ((1-self.alpha)*(np.dot(r_s.T, r_s).ravel() / 2. + self.alpha*np.dot(r_t.T, r_t).ravel() / 2.  - r_t.sum(axis=0)) /N_min).item()  # LOOCV
                aux_params={"sigma":sigma_list[s],"gamma":g}
                j_scores_[str(aux_params)]=-1*score
               
                if verbose:
                    print("Parameters %s -- J-score = %.3f"% (str(aux_params),score))
                if j_scores_[str(aux_params)] > max_:
                   best_params_ = aux_params
                   max_ = j_scores_[str(aux_params)]  
                   
        kernel_=Gauss_Kernel(dictionary=kernel.dictionary,gamma=best_params_["sigma"])
                
        return  kernel_,best_params_["gamma"]
        
           
    def PE_divergence(self,data_ref=None,data_test=None):
    ####### Function estimating the Pearson Divergence 

    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution q_v(.)

    ## Output 
    # score: Pearson Divergence
          
        phi_test=self.kernel.k_V(data_test)
        phi_ref=self.kernel.k_V(data_ref)
  
        N_test=len(phi_test)
        N_ref=len(phi_ref)
       
        H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+self.gamma*np.eye(self.kernel.n), h)

        score=(theta.transpose()).dot(H).dot(theta)
        score*=-0.5
        score+=theta.transpose().dot(h)
        score-=0.5 
        return score
    
    def r_(self,theta,data):
        
        ### Likelihood-ratio function     
        ##### Input
        # theta: estimated parameter
        # data: datapoints to evaluate in the likelihood ratios     
        #### Ouput: 
        # ratio:estimated likelihood ratio at the data points data 

        aux_theta=1*theta
        phi=self.kernel.k_V(transform_data(data))
        ratio=phi.dot(theta)
 
        return ratio
    
class RULSIF_nodes():
    def __init__(self,data_ref,data_test,alpha):

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
    
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


        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.N_ref=len(data_ref[0])
        self.N_test=len(data_test[0])
        self.n_nodes=len(data_ref)
        self.alpha=alpha
        
        self.RULSIF_models=[]
        
        for i in range(self.n_nodes):
            self.RULSIF_models.append(RULSIF(self.data_ref[i],self.data_test[i],alpha=self.alpha))
            
    def fit(self,data_ref=None,data_test=None): 
    ### Function estimating the theta parameter associated with the likelihood ratios 
    ### for a given set of observations coming from p_v(.) and q_v(.) 

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
              
        theta=[]
        for i in range(len(self.RULSIF_models)):
            theta.append(1.*self.RULSIF_models[i].fit(data_ref[i],data_test[i]))
           
        return theta
   
    def r_v(self,node_index,theta,data_node):
        
   ### Estimate likelihood-ratio function at node node_index

   ## Input
   # node_index: index of the node where the likelihood ratio is evaluated.
   # theta: estimated parameter.
   # data: data points used to evaluate the likelihood ratio at node v.
   
   ## Output
   # Likelihood ratio at node node_index evaluated at the points in data.
        
        ratio=self.RULSIF_models[node_index].r_(theta[node_index],data_node)

        return ratio
    
    def update_hs(self,data_ref=None,data_test=None):
    ### This function estimates the variables H_ref, H_test, and h_test for each parameter in the estimation of RULSIF.

## Input
# data_ref and data_test: lists with n_nodes elements, where each element is a numpy array of dimension (N, L), 
#                         where N is the size of the dataset and L is the dimension of the feature space.
# data_ref: data points representing the distribution p_v(.).
# data_test: data points representing the distribution q_v(.).

## Output
# H_ref: list of matrices containing the means of K() K()^T over observations from the distribution p_v(.).
# H_test: list of matrices containing the means of K() K()^T over observations from the distribution q_v(.).
# h_test: list of vectors containing the means of K() over observations from the distribution q_v(.).

    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
                
        phi_test=[]
        phi_ref=[]  
        
          
        for i in range(self.n_nodes):
            phi_test.append(self.RULSIF_models[i].kernel.k_V(data_test[i]))
            phi_ref.append(self.RULSIF_models[i].kernel.k_V(data_ref[i]))
            
    ############# Estimate H
        N_ref=len(phi_ref[0])
        N_test=len(phi_test[0])
        h_test =[[] for i in range(self.n_nodes)]
        H_ref=[[] for i in range(self.n_nodes)]
        H_test=[[] for i in range(self.n_nodes)]
        for i in range(self.n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H_ref[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H_test[i]=np.einsum('ji,j...',  phi_test[i],phi_test[i])
            H_ref[i]/=N_ref 
            H_test[i]/=N_test
            
        return H_ref,H_test,h_test
    
    def PE_divergence(self,theta,data_ref=None,data_test=None):
        
    ### Function estimating the Pearson Divergence at the node level 

    ## Input
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v, n'_v, L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output 
    # score: numpy vector of dimension n_nodes; each element represents the Pearson Divergence at the node level.
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
        
        H_ref,H_test,h_test=self.update_hs(data_ref,data_test)
        score=np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            score[i]=(theta[i].T).dot(H_test[i]).dot(theta[i])*self.alpha
            score[i]+=(theta[i].T).dot(H_ref[i]).dot(theta[i])*(1-self.alpha)
            score[i]*=-0.5
            score[i]+=(theta[i].T).dot(h_test[i])
            score[i]-=0.5 
        return score
    
    

class ULSIF():
########## Class implementing the ULSIF f-divergence estimator
    def __init__(self,data_ref,data_test):     
    
    ## Input

    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution p_v'(.) 
        
        self.data_ref=transform_data(data_ref)
        self.data_test=transform_data(data_test)
        kernel_1=self.initializalize_kernel() 
        self.kernel,self.gamma=self.model_selection(kernel_1)

     
    def initializalize_kernel(self):
 
    ## Output
    # kernel_1: A initialized kernel with a given dictionary 
        if self.data_test.shape[0]>100:
            index=np.random.choice(len(self.data_test),replace=False,size=100)
            dictionary=1*self.data_test[index]
        else:
            dictionary=1*self.data_test
              
        kernel_1=Gauss_Kernel(dictionary,gamma=1.0)
                   
        return kernel_1
    
    def fit(self,data_ref=None,data_test=None,kernel=None,gamma=None): 
#### Function estimating the theta parameter for a given set of observations coming from p_v(.) and q_v(.) 

## Input
# data_ref: data points representing the distribution p_v(.).
# data_test: data points representing the distribution q_v(.).
# kernel: kernel to be used in the method.
# gamma: penalization constant related to the sparsity of the parameters.

## Output
# theta: the parameter related to likelihood-ratio estimation.

     
        if  data_ref is not None:
            data_ref=transform_data(data_ref)
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=transform_data(data_test)
        else:
            data_test=self.data_test
    
        if kernel is None:
            kernel=self.kernel
        
        if gamma is None:
            gamma=self.gamma
    
        n_centers=kernel.n
        phi_test=kernel.k_V(data_test)
        phi_ref=kernel.k_V(data_ref)
   
        N_test=len(phi_test)
        N_ref=len(phi_ref)
        
        H=np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+gamma*np.eye(n_centers), h)

        return theta
    
    def model_selection(self,kernel,verbose=False):

        ###### Input
        # kernel: kernel to be used in the method
        # verbose: whether or not to print intermediate steps 
        
        ###### Output
        # kernel_,gamma: kernel initialized with the selected dictionary and the optimal value of gamma
        
        
        sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])
        max_ = -np.inf
        j_scores_ = {}
        best_params_={}
        N_test=self.data_test.shape[0]
        N_ref=self.data_ref.shape[0]
        N_min=np.min((N_ref,N_test))
        gamma_list = np.logspace(-5,1,4)
        
        if N_ref<N_test:
            index_data = np.random.choice(
                            N_test,
                            N_ref,
                            replace=False)
        elif N_test<N_ref:
            index_data = np.random.choice(
                            N_ref,
                            N_test,
                            replace=False)
            
        
        for s in range(len(sigma_list)):
            kernel_=Gauss_Kernel(dictionary=kernel.dictionary,gamma=sigma_list[s])

            
            if N_ref<N_test:
                phi_test = kernel_.k_V(self.data_test[index_data])
                phi_ref  = kernel_.k_V(self.data_ref)
                
            elif N_test<N_ref:
                phi_test = kernel_.k_V(self.data_test)
                phi_ref =  kernel_.k_V(self.data_ref[index_data])
            else:
                phi_test =  kernel_.k_V(self.data_test)
                phi_ref =   kernel_.k_V(self.data_ref)
            
        
            H=np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref 
         
            h = np.mean(phi_test, axis=0)
          
            h = h.reshape(-1, 1)
            for g in gamma_list:
                B = H + np.identity(kernel.n) * (g * (N_test - 1) / N_test)
                BinvX = np.linalg.solve(B, phi_test.T)
                XBinvX = phi_test.T * BinvX
                D0 = np.ones(N_min) * N_test- np.dot(np.ones(kernel.n), XBinvX)
                diag_D0 = np.diag((np.dot(h.T, BinvX) / D0).ravel())
                B0 = np.linalg.solve(B, h * np.ones(N_min)) + np.dot(BinvX, diag_D0)
                diag_D1 = np.diag(np.dot(np.ones(kernel.n), phi_ref.T * BinvX).ravel())
                B1 = np.linalg.solve(B,  phi_ref.T) + np.dot(BinvX, diag_D1)
                B2 = (N_test- 1) * (N_ref* B0 - B1) / (N_test* (N_ref - 1))
                B2[B2<0]=0
                r_s = (phi_ref.T * B2).sum(axis=0).T
                r_t= (phi_test.T * B2).sum(axis=0).T   
                score = ((np.dot(r_s.T, r_s).ravel() / 2.  - r_t.sum(axis=0)) /N_min).item()  # LOOCV
                aux_params={"sigma":sigma_list[s],"gamma":g}
                j_scores_[str(aux_params)]=-1*score
               
                if verbose:
                    print("Parameters %s -- J-score = %.3f"% (str(aux_params),score))
                if j_scores_[str(aux_params)] > max_:
                   best_params_ = aux_params
                   max_ = j_scores_[str(aux_params)]  
                   
        kernel_=Gauss_Kernel(dictionary=kernel.dictionary,gamma=best_params_["sigma"])
                
        return  kernel_,best_params_["gamma"]
        
           
    def PE_divergence(self,data_ref=None,data_test=None):
    ####### Function estimating the Pearson Divergence 

    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution q_v(.)

    ## Output 
    # score: Pearson Divergence 
          
          
        phi_test=self.kernel.k_V(data_test)
        phi_ref=self.kernel.k_V(data_ref)
  
        N_test=len(phi_test)
        N_ref=len(phi_ref)
       
        H=np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+self.gamma*np.eye(self.kernel.n), h)

        score=(theta.transpose()).dot(H).dot(theta)
        score*=-0.5
        score+=theta.transpose().dot(h)
        score-=0.5 
        return score
    
    def r_(self,theta,data):
        ### Likelihood-ratio function     
        ##### Input
        # theta: estimated parameter
        # data: datapoints to evaluate in the likelihood ratios     
        #### Ouput: 
        # ratio:estimated likelihood ratio at the data points data 

        phi=self.kernel.k_V(transform_data(data))
        ratio=phi.dot(theta)
 
        return ratio

class ULSIF_nodes():
    def __init__(self,data_ref,data_test):
    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.

        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
           
        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.N_ref=len(self.data_ref[0])
        self.N_test=len(self.data_test[0])
        self.n_nodes=len(self.data_ref)
  
        self.ULSIF_models=[]
        
        for i in range(self.n_nodes):
          #  print("Fitting RULSIF model for node %i"%i)
            self.ULSIF_models.append(ULSIF(self.data_ref[i],self.data_test[i]))
            
    def fit(self,data_ref=None,data_test=None): 
    ### Function estimating the theta parameter associated with the likelihood ratios 
    ### for a given set of observations coming from p_v(.) and q_v(.) 

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
                
        theta=[]
        for i in range(len(self.ULSIF_models)):
            theta.append(1.*self.ULSIF_models[i].fit(data_ref[i],data_test[i]))
           
        return theta
   
    def r_v(self,node_index,theta,data_node):
        
   ### Estimate likelihood-ratio function at node node_index

   ## Input
   # node_index: index of the node where the likelihood ratio is evaluated.
   # theta: estimated parameter.
   # data: data points used to evaluate the likelihood ratio at node v.
   
   ## Output
   # Likelihood ratio at node node_index evaluated at the points in data.
        
        ratio=self.ULSIF_models[node_index].r_(theta[node_index],data_node)

        return ratio
    
    def update_hs(self,data_ref=None,data_test=None):
    ### This function estimates the variables H_ref, H_test, and h_test for each parameter in the estimation of RULSIF.

## Input
# data_ref and data_test: lists with n_nodes elements, where each element is a numpy array of dimension (N, L), 
#                         where N is the size of the dataset and L is the dimension of the feature space.
# data_ref: data points representing the distribution p_v(.).
# data_test: data points representing the distribution q_v(.).

## Output
# H_ref: list of matrices containing the means of K() K()^T over observations from the distribution p_v(.).
# h_test: list of vectors containing the means of K() over observations from the distribution q_v(.).

    
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
                
        phi_test=[]
        phi_ref=[]  
        
          
        for i in range(self.n_nodes):
            phi_test.append(self.ULSIF_models[i].kernel.k_V(data_test[i]))
            phi_ref.append(self.ULSIF_models[i].kernel.k_V(data_ref[i]))
            
  
    ############# Estimate H
        N_ref=len(phi_ref[0])
        N_test=len(phi_test[0])
        h_test =[[] for i in range(self.n_nodes)]
        H_ref=[[] for i in range(self.n_nodes)]
        for i in range(self.n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i]/= N_test
            H_ref[i]=np.einsum('ji,j...',  phi_ref[i],phi_ref[i])
            H_ref[i]/=N_ref 
    
        return H_ref,h_test
    
    def PE_divergence(self,theta,data_ref=None,data_test=None):     
       ### Function estimating the Pearson Divergence at the node level 

       ## Input
       # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
       # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v, n'_v, L), 
       #                         where L is the dimension of the feature space.
       # data_ref[v]: data points representing the distribution p_v(.).
       # data_test[v]: data points representing the distribution q_v(.).

       ## Output 
       # score: numpy vector of dimension n_nodes; each element represents the Pearson Divergence at the node level. 
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
        
        H_ref,h_test=self.update_hs(data_ref,data_test)
        score=np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            score[i]=(theta[i].T).dot(H_ref[i]).dot(theta[i])
            score[i]*=-0.5
            score[i]+=(theta[i].T).dot(h_test[i])
            score[i]-=0.5 
        return score    
    
 
    

class KLIEP():
########## Class implementing the GULSIF f-divergence estimation 
    def __init__(self,data_ref,data_test,k_cross_validation=5,tol=1e-2,lr=1e-4):     
    
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution p_v'(.) 
    # tol: level of accepted tolerence in the estimation
    # k_cross_validation: number of splits to do for cross validation   
    # lr: learnin rate associated with the optimization problem
        
        self.data_ref=transform_data(data_ref)
        self.data_test=transform_data(data_test)
        self.tol=tol
        self.k_cross_validation=k_cross_validation
        self.lr=lr
        kernel_1=self.initializalize_kernel() 
        self.kernel=self.model_selection(kernel_1,verbose=False)

     
    def initializalize_kernel(self):
 
    ## Output
    # kernel_1: A initialized kernel with a given dictionary 
     
        if self.data_test.shape[0]>100:
            index=np.random.choice(len(self.data_test),replace=False,size=100)           
            dictionary=1*self.data_test[index]
        else:
            dictionary=1*self.data_test
            
        kernel_1=Gauss_Kernel(dictionary,gamma=1.0)
                   
        return kernel_1
    
    def fit(self,data_ref=None,data_test=None,kernel=None,verbose=False): 
    #### Function estimating the theta parameter for a given set of observations comming from p_v(.) and q_v(.) 
    
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution q_v(.) 
    # kernel: kernel to be used in the method
    # verbose: whether or not to print the convergence bahaviour of the estimates
    
    ## Output
    # theta: the parameter related with likeliood-ratio estimation 
    
        if  data_ref is not None:
            data_ref=transform_data(data_ref)
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=transform_data(data_test)
        else:
            data_test=self.data_test
    
        if kernel is None:
            kernel=self.kernel

        EPS=1e-6
    
        phi_test = kernel.k_V(data_test)
        phi_ref= kernel.k_V(data_ref)
        lr=copy.deepcopy(self.lr)
        
        b= np.mean(phi_ref, axis=0)+EPS
        b = b.reshape(-1, 1)

        theta= np.ones((kernel.n, 1)) / kernel.n
        previous_objective = -np.inf
        objective = np.mean(np.log(np.dot(phi_test, theta) + EPS))
        if verbose:
                print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
        k = 0
        while objective-previous_objective > self.tol:
            if previous_objective<objective:
                lr/=100
            previous_objective = objective
            theta_p = np.copy(theta)
            r=1./np.clip(np.dot(phi_test, theta),EPS,np.inf)
            theta += lr * np.dot(np.transpose(phi_test),r)
            
            theta += b * ((((1-np.dot(np.transpose(b), theta)) /
                            (np.dot(np.transpose(b), b) + EPS))))
            theta = np.maximum(0, theta)
            den=np.dot(np.transpose(b), theta) + EPS
            theta /= den
           
           # theta=self._projection_PG(theta, b).reshape(-1,1)
            logs_=np.log(np.dot(phi_test, theta) + EPS)
            objective = np.mean(logs_)
            k += 1
            
            if verbose:
                if k%100 == 0:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))

        return theta
    

    def model_selection(self,kernel,verbose=False):
               
        ###### Input
        # kernel: kernel to be used in the method
        # verbose: whether or not to print intermediate steps 
        
        ###### Output
        # kernel_: kernel initialized with the selected dictionary and the optimal value of alpha
 
        sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])
        max_ = -np.inf
        j_scores_ = {}
        best_params_={}
        N_test=self.data_test.shape[0]
        N_ref=self.data_ref.shape[0]
   
        ref_index_validation=[np.arange(N_ref)[int(i*(N_ref/self.k_cross_validation)):int((i+1)*((N_ref/self.k_cross_validation)))] for i in range(self.k_cross_validation)]
        test_index_validation=[np.arange(N_test)[int(i*(N_test/self.k_cross_validation)):int((i+1)*((N_test/self.k_cross_validation)))] for i in range(self.k_cross_validation)]
        ref_index_train=[]
        test_index_train=[]
        
        for i in range(self.k_cross_validation):
            if i==0:
                ref_index_train.append(np.hstack(ref_index_validation[i+1:]))
                test_index_train.append(np.hstack(test_index_validation[i+1:]))
            elif i==(self.k_cross_validation-1):
                ref_index_train.append(np.hstack(ref_index_validation[:i]))
                test_index_train.append(np.hstack(test_index_validation[:i]))
            else:
                ref_index_train.append(np.hstack((np.hstack(ref_index_validation[:i]),np.hstack(ref_index_validation[i+1:]))))
                test_index_train.append(np.hstack((np.hstack(test_index_validation[:i]),np.hstack(test_index_validation[i+1:]))))


        cost_vector=np.zeros(len(sigma_list))
        for s in range(len(sigma_list)):
            kernel_=Gauss_Kernel(kernel.dictionary,gamma=sigma_list[s])
            aux_cost=np.zeros(self.k_cross_validation)
           
            for i in range(self.k_cross_validation):
                theta=self.fit(self.data_ref[ref_index_train[i]],self.data_test[test_index_train[i]],kernel=kernel_,verbose=False)
                phi=kernel.k_V(self.data_test[test_index_validation[i]])
                aux_cost[i]=np.mean(np.log(phi.dot(theta)+1e-6))
            
            aux_params={"sigma":sigma_list[s]}
            j_scores_[str(aux_params)]=np.mean(aux_cost)
            if verbose:
                print("Parameters %s -- J-score = %.3f"% (str(aux_params), j_scores_[str(aux_params)]))
            if j_scores_[str(aux_params)] > max_:
                best_params_ = aux_params
                max_ = j_scores_[str(aux_params)]  
                   
        kernel_=Gauss_Kernel(dictionary=kernel.dictionary,gamma=best_params_["sigma"])
                
        return  kernel_
           
    def KL_divergence(self,data_ref=None,data_test=None):
    ####### Function estimating the Kullback–Leibler Divergence 

    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution q_v(.)

    ## Output 
    # score: Kullback–Leibler Divergence
          
    
        data_ref=transform_data(data_ref)
        data_test=transform_data(data_test)  
        
        theta=self.fit(data_ref,data_test)
        
        score=np.mean(np.log(self.r_(theta,data_test)+1e-6))
        
        return score
    
    def r_(self,theta,data):
        ######## Likelihood ratio estimation 
        
        ##### Input
        # theta: estimated parameter
        # data: datapoints to evaluate in the likelihood ratios     
        #### Ouput: 
        # estimated likelihood ratio at the data points data 

        phi=self.kernel.k_V(transform_data(data))
        ratio=phi.dot(theta)
 
        return ratio


class KLIEP_nodes():
    def __init__(self,data_ref,data_test):
    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).
    # alpha: regularization parameter associated with the upper bound of the likelihood ratio.

        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==n_nodes_2):
                raise ValueError(F"The datasets should be list and have the same number of elements")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)            

        self.data_ref=[transform_data(d) for d in data_ref]
        self.data_test=[transform_data(d) for d in data_test]
        self.N_ref=len(data_ref[0])
        self.N_test=len(data_test[0])
        self.n_nodes=len(data_ref)
        self.KLIEP_models=[]
        
        for i in range(self.n_nodes):
            self.KLIEP_models.append(KLIEP(self.data_ref[i],self.data_test[i]))
            
    def fit(self,data_ref=None,data_test=None): 
    ### Function estimating the theta parameter associated with the likelihood ratios 
    ### for a given set of observations coming from p_v(.) and q_v(.) 

    ## Input
    # data_ref and data_test: lists with n_nodes elements, each containing a numpy array of dimension (n_v(n'_v), L), 
    #                         where L is the dimension of the feature space.
    # data_ref[v]: data points representing the distribution p_v(.).
    # data_test[v]: data points representing the distribution q_v(.).

    ## Output
    # theta: numpy matrix of dimension n_nodes x L; each row is the estimated parameter for the node v.
    
    
        if data_ref is not None:
            data_ref=[transform_data(d) for d in data_ref]
        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
            
        if data_ref is None or data_test is None:
            if data_ref is None:
                data_ref=self.data_ref
            if data_test is None:
                data_test=self.data_test
        theta=[]
        for i in range(len(self.KLIEP_models)):
            theta.append(1.*self.KLIEP_models[i].fit(data_ref[i],data_test[i]))
           
        return theta
   
    def r_v(self,node_index,theta,data_node):
        
        
        ######## Likelihood ratio estimation 
        
        ##### Input
        # node_index: likelihood ratio to be evaluated 
        # theta: estimated parameter
        # data:_node datapoints to evaluate in the likelihood ratios at node v
        
        ## Output
        ## Likelihood_ratio at node node_index evaluated at the points data_node. 
        
        
        ratio=self.KLIEP_models[node_index].r_(theta[node_index],data_node)
        return ratio
   
    
    def KL_divergence(self,theta,data_test=None):   

    ####### Function estimating the Kullback–Leibler Divergence at the node level 

    ## Input
    ## theta: list of node_level parameters associated to the likelihood-ratio
    # data_test: data points representing the distribution q_v(.) for all the nodes

    ## Output 
    # score: Pearson Divergence at the node level 
    

        if  data_test is not None:   
            data_test=[transform_data(d) for d in data_test]
        else:
            data_test=self.data_test
       
        score=np.zeros(len(theta))
        for i in range(self.n_nodes):
            score[i]=np.mean(np.log(self.KLIEP_models[i].r_(theta[i],data_test[i])+1e-6))
        return score
            

    
    
    
    
    