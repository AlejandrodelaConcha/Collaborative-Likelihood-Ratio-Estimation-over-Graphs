# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  aux_functions
# Author(s):  
# Initial version:  2024-01-15
# Last modified:    2025-02-25              
# Current version:  2025-02-25
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to define functions associated with the Gaussian Kernel and the Nystrom Approximation.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy, numba, scipy, joblib, dill
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Gaussian Kernel, Laplacian Kernel, Kernel Function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
from numba import njit,jit
from itertools import product
from scipy.sparse import csr_matrix
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import dill as pickle

def transform_data(x):
    ### Function taking care of addresing possible problems with the numpy array 
    if len(x.shape)==1:
        return np.atleast_2d(x.astype(np.float64)).T
    else:
        return np.atleast_2d(x.astype(np.float64))


@jit(nopython=True)
def calc_dist(A,B,sqrt=False):
   ######### This function is a fast version of the distance between elements of two matrices 
   
   #### Input
   ## A: nxp numpy array of n elements of dimension p
   ## B: nxp numpy array of m elements of dimension p 
   ## sqrt: whether the square of the distance of the distance is reported
   
   #### Output 
   ### dist: n*m matrix containing the distance or its squared between the elements of A and B 
    
  dist=np.dot(A,B.T)

  TMP_A=np.empty(A.shape[0],dtype=A.dtype)
  for i in range(A.shape[0]):
    sum=0.
    for j in range(A.shape[1]):
      sum+=A[i,j]**2
    TMP_A[i]=sum

  TMP_B=np.empty(B.shape[0],dtype=A.dtype)
  for i in range(B.shape[0]):
    sum=0.
    for j in range(B.shape[1]):
      sum+=B[i,j]**2
    TMP_B[i]=sum

  if sqrt==True:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=np.sqrt(-2.*dist[i,j]+TMP_A[i]+TMP_B[j])
  else:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=-2.*dist[i,j]+TMP_A[i]+TMP_B[j]
  return dist


@jit(nopython=True)
def calc_dist_L1(A,B):
    #### Input
    ## A: nxp numpy array of n elements of dimension p
    ## B: nxp numpy array of m elements of dimension p 
    
    #### Output 
    ### dist: n*m matrix containing the L_1 distance between the elements of A and B 
    
    N=A.shape[0]
    M=B.shape[0]
    dist=np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist[i,j]=np.sum(np.abs(A[i]-B[j]))  
    return dist
    


class Gauss_Kernel(object):
#### Class implementing the Gaussian kernel
    def __init__(self,dictionary,gamma):
    ## Input
    # dictionary: An initial dictionary
    # gamma: Width parameter 
        self.dictionary=dictionary
        self.gamma=gamma
        self.n=self.dictionary.shape[0]
        self.d=self.dictionary.shape[1]

    def k(self,x):
    ### Function estimating the feature transformation of point x with respect to a given dictionary
    ## Input
    # x: point to evaluate 
    ## output
    # gaussian kernel evaluated at x
    
        if self.n==1:
            distances=np.linalg.norm(x-self.dictionary)**2
            distances*=-(self.gamma/self.d)
            distances=np.atleast_2d(np.array(np.exp(distances),dtype=np.float64))
            return distances
        else:       
            distances=calc_dist(np.atleast_2d(x),self.dictionary)
            distances*=-(self.gamma/self.d)
            return np.exp(distances)


    def k_V(self,x):
    ### Function estimating the feature transformation of a set of points with respect to a given dictionary
    ## Input
    # x: point to evaluate 
    ## Output 
    # n np array representing the kernel evaluating at each point on the set x
    
        x=transform_data(x)        
        if len(x)==1:
            return self.k(x)           
        else:         
            distances= calc_dist(x,self.dictionary)
            distances*=-(self.gamma/self.d)
            return np.exp(distances)
  
    def coherence(self,x): 
    ### Function implementing the coherence of a point with respect to a given dictionary 
    ## Input
    # x: point to evaluate 
    ## output.
    # k_: the empirical feature map of x with respect to the available dictionary 
    # mu_0: coherence of the algorithm 
    
        k_=self.k(x)       
        mu_0=np.max(np.abs(k_))       
        return k_,mu_0
 
    def add_dictionary(self,x):   
    ##### Function to add a points to the dictionary
    ## Input
    # x: point to add 
        self.dictionary=np.vstack((self.dictionary,x))
        self.n=len( self.dictionary)
        
    def new_phi(self,data,new_value):
    ##### This function k(x,new_value) where x is a set of observations
    ### Input
    ## data: the data to be evaluated against new value
    ## new value: the point used as reference.
    ### Output: 
    ## the matrix k(data,new_value)
        data=transform_data(data)
        new_value=np.atleast_2d(new_value)
        distances= calc_dist(new_value,data)
        distances*=-(self.gamma/self.d)
        return np.exp(distances)
    
    def get_internal_coherences(self):
    ### This function computes the coherence of an element of the dictionary with respect other points the dictionary
    ### Output: 
    ## coherence= the coherence of eact element of the dictionary with respect to the others.
        coherences=[]
        aux_coherence=self.k(self.dictionary[0])[0]
        coherences.append(np.max(np.abs(aux_coherence[1:])))
        for i in range(len(self.dictionary)-1):
            aux_coherence=self.k(self.dictionary[i])[0]
            coherences.append(np.max(np.abs(np.hstack((aux_coherence[:i],aux_coherence[i+1:])))))
        i=len(self.dictionary)-1
        aux_coherence=self.k(self.dictionary[i])[0]
        coherences.append(np.max(np.abs(np.array((aux_coherence[:i])))))
        coherences=np.array(coherences)
        return coherences
    
  
    

class Laplace_Kernel(object):
#### Class implementing the Laplacian kernel
    def __init__(self,dictionary,gamma):
    ## Input
    # dictionary: An initial dictionary
    # gamma: Width parameter 
        self.dictionary=dictionary
        self.gamma=gamma
        self.n=self.dictionary.shape[0]
        self.d=self.dictionary.shape[1]
        

    def k(self,x):
    ### Function estimating the feature transformation of point x with respect to a given dictionary
    ## Input
    # x: point to evaluate 
    ## output
    # laplace kernel evaluated at x  
        if self.n==1:
            distances=np.sum(np.abs(x-self.dictionary),axis=1)
            distances*=-(self.gamma/self.d)
            distances=np.atleast_2d(np.array(np.exp(distances),dtype=np.float64))
            return distances
        else:       
            distances=calc_dist_L1(np.atleast_2d(x),self.dictionary)
            distances*=-(self.gamma/self.d)
            return np.exp(distances)


    def k_V(self,x):
    ### Function estimating the feature transformation of a set of points with respect to a given dictionary
    ## Input
    # x: point to evaluate 
    ## Output 
    # n np array representing the kernel evaluating at each point on the set x
    
        x=transform_data(x)        
        if len(x)==1:
            return self.k(x)           
        else:         
            distances= calc_dist_L1(x,self.dictionary)
            distances*=-(self.gamma/self.d)
            return np.exp(distances)
  
    def coherence(self,x): 
    ### Function implementing the coherence of a point with respect to a given dictionary 
    ## Input
    # x: point to evaluate 
    ## output.
    # k_: the empirical feature map of x with respect to the available dictionary 
    # mu_0: coherence of the algorithm 
    
        k_=self.k(x)       
        mu_0=np.max(np.abs(k_))       
        return k_,mu_0
 
    def add_dictionary(self,x):   
    ##### Function to add a points to the dictionary
    ## Input
    # x: point to add 
    
        self.dictionary=np.vstack((self.dictionary,x))
        self.n=len( self.dictionary)
        
    def new_phi(self,data,new_value):
    ##### This function k(x,new_value) where x is a set of observations
    ### Input
    ## data: the data to be evaluated against new value
    ## new value: the point used as reference.
    ### Output: 
    ## the matrix k(data,new_value)
        data=transform_data(data)
        new_value=np.atleast_2d(new_value)
        distances= calc_dist_L1(new_value,data)
        distances*=-(self.gamma/self.d)
        return np.exp(distances)
    
    def get_internal_coherences(self):
    #### This function computes the coherence of an element of the dictionary with respect other points the dictionary 
        coherences=[]
        aux_coherence=self.k(self.dictionary[0])[0]
        coherences.append(np.max(np.abs(aux_coherence[1:])))
        for i in range(len(self.dictionary)-1):
            aux_coherence=self.k(self.dictionary[i])[0]
            coherences.append(np.max(np.abs(np.hstack((aux_coherence[:i],aux_coherence[i+1:])))))
        i=len(self.dictionary)-1
        aux_coherence=self.k(self.dictionary[i])[0]
        coherences.append(np.max(np.abs(np.array((aux_coherence[:i])))))
        coherences=np.array(coherences)
        return coherences
          

@jit(nopython=True)
def product_gaussian(gamma,data_1,data_2):
##### This function evaluates the Gaussian kernel between two sets of observations 
### Input
## gamma: width parameter of the Gaussian Kernel
## data_1: matrix nxd made of n observations of dimension d
## data_2: matrix mxd made of m observations of dimension n
## new value: the point used as reference.
### Output: 
## Matrix of dimension n*m with K_{i,j}=K(x_i,x_j) 
    d=data_1.shape[1]
    distances= calc_dist(data_1,data_2)
    distances*=-gamma/d
    distances=distances+distances.T
    distances/=2
    return np.exp(distances)


@jit(nopython=True)
def product_laplace(gamma,data_1,data_2):
##### This function evaluates the Laplacian Kernel between two sets of observations 
### Input
## gamma: width parameter of the Laplacian Kernel
## data_1: matrix nxd made of n observations of dimension d
## data_2: matrix mxd made of m observations of dimension d
## new value: the point used as reference.
### Output: 
## Matrix of dimension n*m with K_{i,j}=K(x_i,x_j) 
    d=data_1.shape[1]
    distances= calc_dist_L1(data_1,data_2)
    distances*=-gamma/d
    distances=distances+distances.T
    distances/=2
    return np.exp(distances)


class Nystrom_Kernel(object):
#### Class implementing Nystrom approximation of the Main Kernel 
    def __init__(self,dictionary,gamma,kernel_type="Gaussian"):
    ## Input
    # dictionary: An initial dictionary
    # gamma: Width parameter 
    # kernel_type: the type of kernel that is used for stimation 
        self.dictionary=dictionary
        self.gamma=gamma
        self.n=self.dictionary.shape[0]
        self.d=self.dictionary.shape[1]
        self.kernel_type=kernel_type
        self.K=self.product_(self.gamma,self.dictionary,self.dictionary)
        eps=1e-6 ### parameter to garantee numerical stability 
        self.K=self.K+eps*np.eye(self.n)
        self.eigvalues,self.eigvectors=np.linalg.eigh(self.K)
        
    def product_(self,gamma,data_1,data_2):
        ### Function to estimate K(data_1,data_2)
        if self.kernel_type=="Gaussian":
            return product_gaussian(gamma,data_1,data_2)
        elif self.kernel_type=="Laplace":
            return product_laplace(gamma,data_1,data_2)
            
    def k(self,x):
    ### Function estimating the feature transformation of point x with respect to a given dictionary
    ## Input
    # x: point to evaluate 

        if self.kernel_type=="Gaussian":
            if self.n==1:
                distances=np.linalg.norm(x-self.dictionary)**2
                distances*=-(self.gamma/self.d)
                distances=np.atleast_2d(np.array(np.exp(distances),dtype=np.float64))
                k_x= distances.dot(self.eigvectors)
                k_x= k_x.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_x
            else:       
                distances=calc_dist(np.atleast_2d(x),self.dictionary)
                distances*=-(self.gamma/self.d)
                distances=np.exp(distances)
                k_x= distances.dot(self.eigvectors)
                k_x= k_x.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_x
        
            
        if self.kernel_type=="Laplace":
            if self.n==1:
                distances=np.sum(np.abs(x-self.dictionary),axis=0)
                distances*=-(self.gamma/self.d)
                distances=np.atleast_2d(np.array(np.exp(distances),dtype=np.float64))
                k_x= distances.dot(self.eigvectors)
                k_x= k_x.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_x
            else:       
                distances=calc_dist_L1(np.atleast_2d(x),self.dictionary)
                distances*=-(self.gamma/self.d)
                distances=np.exp(distances)
                k_x= distances.dot(self.eigvectors)
                k_x= k_x.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_x


    def k_V(self,x):
    ### Function estimating the feature transformation of a set of points with respect to a given dictionary
    ## Input
    # x: point to evaluate 
        x=transform_data(x)  
        
        if self.kernel_type=="Gaussian":
            if len(x)==1:
                return self.k(x)           
            else:         
                distances= calc_dist(x,self.dictionary)
                distances*=-(self.gamma/self.d)
                distances=np.exp(distances)
                k_v= distances.dot(self.eigvectors)
                k_v= k_v.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_v
        
            
        if self.kernel_type=="Laplace":
            if len(x)==1:
                return self.k(x)           
            else:         
                distances= calc_dist_L1(x,self.dictionary)
                distances*=-(self.gamma/self.d)
                distances=np.exp(distances)
                k_v= distances.dot(self.eigvectors)
                k_v= k_v.dot(np.diag(1/np.sqrt(self.eigvalues))) 
                return k_v
  
    def coherence(self,x): 
    ### Function implementing the coherence of a point with respect to a given dictionary 
    ## Input
    # x: point to evaluate 
    ## output.
    # k_: the empirical feature map of x with respect to the available dictionary 
    # mu_0: coherence of the algorithm 
        k_=self.k(x)       
        mu_0=np.max(np.abs(k_))       
        return k_,mu_0
 
    def add_dictionary(self,x):   
    ##### Function to add a points to the dictionary
    # x: point to add 
        self.dictionary=np.vstack((self.dictionary,x))
        self.n=len( self.dictionary)    
        self.K=self.product_(self.gamma,self.dictionary,self.dictionary)
        eps=1e-6 ### parameter to garantee numerical stability 
        self.K=self.K+eps*np.eye(self.n)
        self.eigvalues,self.eigvectors=np.linalg.eigh(self.K)
    

    
    def k_product(self,x,y):
    ### Function to estimate K(data_1,data_2)
        x=np.atleast_2d(x)
        y=np.atleast_2d(y)
        k_x= self.k_V(x)
        k_y= self.k_V(y)
        product=(k_x).dot(k_y.T)
        return product
    
    def get_internal_coherences(self):
    #### This function computes the coherence of an element of the dictionary with respect other points the dictionary 
        coherences=[]
        aux_coherence=self.k(self.dictionary[0])[0]
        coherences.append(np.max(np.abs(aux_coherence[1:])))
        for i in range(len(self.dictionary)-1):
            aux_coherence=self.k(self.dictionary[i])[0]
            coherences.append(np.max(np.abs(np.hstack((aux_coherence[:i],aux_coherence[i+1:])))))
        i=len(self.dictionary)-1
        aux_coherence=self.k(self.dictionary[i])[0]
        coherences.append(np.max(np.abs(np.array((aux_coherence[:i])))))
        coherences=np.array(coherences)
        return coherences
        
        
  
############################### Functions related to graph manipulation 


def transform_matrix_totime(W, n_times):
    ###### This function generates a multiplex based on a graph W. The algorithm produces n_times copies, 
    ###### where nodes are connected to their "copies" at adjacent times.
    ### Input:
    ## W: adjacency matrix in CSR format.
    ## n_times: the number of time steps considered.
    ### Output:
    ## W_new: adjacency matrix in CSR format representing the multiplex structure.

    
    n_nodes = W.shape[0]
    W_ij = [W.data for n in range(n_times)]
    row_indices = [n*n_nodes+W.row % n_nodes for n in range(n_times)]
    col_indices = [n*n_nodes+W.col % n_nodes for n in range(n_times)]

    row = row_indices[0]
    num_rows = n_nodes
    row_ptr = np.empty(num_rows + 1, dtype=int)
    current_row = 0
    for i in range(len(row)):
        while current_row < row[i]:
            row_ptr[current_row + 1] = i
            current_row += 1
        row_ptr[-1] = len(row)

    W_ij = [W.data for n in range(n_times)]
    row_indices = [n*n_nodes+W.row % n_nodes for n in range(n_times)]
    col_indices = [n*n_nodes+W.col % n_nodes for n in range(n_times)]

    new_row_indices = []
    new_col_indices = []
    new_data = []

    for time in range(n_times-1):
        aux_row_index = 1.*row_indices[time]
        aux_col_index = 1.*col_indices[time]
        aux_data = 1.*W_ij[0]

        aux_row_index = np.hstack((aux_row_index, -1*np.ones(n_nodes)))
        aux_col_index = np.hstack((aux_col_index, -1*np.ones(n_nodes)))
        aux_data = np.hstack((aux_data, -1*np.ones(n_nodes)))

        for i in range(len(row_ptr)-1):
            aux_col_index[row_ptr[i+1]+i:] = np.hstack(
                (-1, aux_col_index[row_ptr[i+1]+i:len(aux_col_index)-1]))
            aux_data[row_ptr[i+1] +
                     i:] = np.hstack((-1, aux_data[row_ptr[i+1]+i:len(aux_data)-1]))
            aux_row_index[row_ptr[i+1]+i:] = np.hstack(
                (-1, aux_row_index[row_ptr[i+1]+i:len(aux_col_index)-1]))
            aux_row_index[row_ptr[i+1]+i] = aux_row_index[row_ptr[i+1]+i-1]
            aux_col_index[row_ptr[i+1] +
                          i] = int((time+1)*n_nodes+aux_row_index[row_ptr[i+1]+i-1] % n_nodes)
            aux_data[row_ptr[i+1]+i] = 1.

        new_row_indices.append(aux_row_index)
        new_col_indices.append(aux_col_index)
        new_data.append(aux_data)

    new_row_indices = np.hstack(
        (np.vstack(new_row_indices).flatten(), row_indices[time+1].flatten()))
    new_col_indices = np.hstack(
        (np.vstack(new_col_indices).flatten(), col_indices[time+1].flatten()))
    new_data = np.hstack(
        (np.vstack(new_data).flatten(), W_ij[time+1].flatten()))

    W_new = csr_matrix((new_data, (new_row_indices, new_col_indices)), shape=(
        n_nodes*n_times, n_nodes*n_times))
    W_new[new_col_indices, new_row_indices] = W_new[new_row_indices, new_col_indices]

    return W_new



def get_componentes(nodes_in,adjacency):
    #### Find the clusters formed by nodes when a boolean condition is satisfied.
    ## Input:
    # nodes_in: nodes that satisfy the condition.
    # adjacency: adjacency matrix.
    ## Output:
    # clusters: list of clusters.

    
    adjacency=adjacency.tocoo()
    mask = np.logical_and(nodes_in[adjacency.row],nodes_in[adjacency.col])
    data = adjacency.data[mask]
    row = adjacency.row[mask]
    col = adjacency.col[mask]
    shape = adjacency.shape
    idx = np.where(nodes_in)[0]
    row = np.concatenate((row, idx))
    col = np.concatenate((col, idx))
    data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
    adjacency = sparse.coo_matrix((data, (row, col)), shape=shape)
    _, components = connected_components(adjacency)


    start = np.min(components)
    stop = np.max(components)
    comp_list = [list() for i in range(start, stop + 1, 1)]
    mask = np.zeros(len(comp_list), dtype=bool)
    for ii, comp in enumerate(components):
        comp_list[comp].append(ii)
        mask[comp] += nodes_in[ii]
    clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
    return clusters



      












