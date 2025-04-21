# -----------------------------------------------------------------------------------------------------------------
# Title:  experiments_LRE
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2024-02-26             
# This version:     2024-02-26
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to generate the experiments presented in the paper 
#               "Collaborative Likelihood-Ratio Estimation over Graphs."
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# numpy, scipy, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GRULSIF, RULSIF, ULSIF, KLIEP, Pool, likelihood-ratio estimation, collaborative likelihood-ratio estimation.
# -----------------------------------------------------------------------------------------------------------------


import numpy as np
from scipy.stats import multivariate_normal,norm,uniform
from scipy.special import logsumexp
import scipy
from pygsp import graphs, filters, plotting
import itertools
import pickle


def generate_experiment_1_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01):
    
    ######### This function implements the experiment 1A of the paper.

    ####### Input
    # n_nodes: number of nodes in the graph.
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # p_within: probability of connection between nodes within the same cluster.
    # p_between: probability of connection between nodes from different clusters.

    ###### Output 
    # G: the graph component of the problem.
    # data_ref: a list of n_nodes elements, each representing observations coming from p_v.
    # data_test: a list of n_nodes elements, each representing observations coming from q_v.
    # affected_nodes: a numpy array of the nodes affected during the experiment.

    
    n_clusters=4
    sample_size=int(n_nodes/n_clusters)
    z=[[i]*int(n_nodes/n_clusters) for i in range(0,n_clusters)]
    z.append([3]*(n_nodes%n_clusters))
    z=list(itertools.chain.from_iterable(z))
    G=graphs.StochasticBlockModel(N=n_nodes,k=n_clusters,z=z,q=p_between,seed=0,p=p_within) 
    
    data_ref=[np.random.normal(size=N_ref) for i in range(n_nodes)]
    data_test=[np.random.normal(size=N_test) for i in range(n_nodes)]

    for i in np.arange(sample_size):
        data_test[i]=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=N_test)
    
    for i in np.arange(3*sample_size,4*sample_size):
        data_test[i]=np.random.normal(loc=1,size=N_test)
        
    affected_nodes=np.hstack((np.arange(sample_size), np.arange(3*sample_size,4*sample_size)))
        
    return G,data_ref,data_test,affected_nodes

def generate_experiment_2_LRE(n_nodes,N_ref,N_test,p_within=0.5,p_between=0.01):
    
    ######### This function implements the experiment 1B of the paper.

    ####### Input
    # n_nodes: number of nodes in the graph.
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # p_within: probability of connection between nodes within the same cluster.
    # p_between: probability of connection between nodes from different clusters.

    ###### Output 
    # G: the graph component of the problem.
    # data_ref: a list of n_nodes elements, each representing observations coming from p_v.
    # data_test: a list of n_nodes elements, each representing observations coming from q_v.
    # affected_nodes: a numpy array of the nodes affected during the experiment.

    

    n_clusters=4
    sample_size=int(n_nodes/n_clusters)
    z=[[i]*int(n_nodes/n_clusters) for i in range(0,n_clusters)]
    z=list(itertools.chain.from_iterable(z))
    G=graphs.StochasticBlockModel(N=n_nodes,k=n_clusters,z=z,q=p_between,seed=0,p=p_within) 
    
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
    
    for i in range(sample_size):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_1,size=N_ref)) 
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_1,size=N_test))  
        
    for i in range(sample_size,2*sample_size):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_2,size=N_ref)) 
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_2,size=N_test))
        
    for i in range(2*sample_size,3*sample_size):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_3,size=N_ref))
        data_test.append(multivariate_normal.rvs(mean=mean,cov=sigma_2_3,size=N_test))
        
    for i in range(3*sample_size,4*sample_size):
        data_ref.append(multivariate_normal.rvs(mean=mean,cov=sigma_1_4,size=N_ref))
        data_test.append(multivariate_normal.rvs(mean=mu_4,cov=sigma_1_4,size=N_test))
   
    affected_nodes=np.hstack((np.arange(2*sample_size,3*sample_size),
                              np.arange(3*sample_size,4*sample_size)))
    return G,data_ref,data_test,affected_nodes
 
def generate_experiment_3_LRE(n_nodes,N_ref,N_test):
    
    ######### This function implements the experiment 2A of the paper.

    ####### Input
    # n_nodes: number of nodes in the graph.
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.

    ###### Output 
    # G: the graph component of the problem.
    # data_ref: a list of n_nodes elements, each representing observations coming from p_v.
    # data_test: a list of n_nodes elements, each representing observations coming from q_v.
    # affected_nodes: a numpy array of the nodes affected during the experiment.

    
    N_1=int(np.sqrt(n_nodes))
    N_2=int(np.ceil(n_nodes/N_1))
    
    G=graphs.Grid2d(N1=N_1,N2=N_2)
    
    affected_nodes=np.arange(N_1*N_2)
    n_nodes=int(N_1*N_2)
    
    A=np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2)],[np.sin(-np.pi/2),np.cos(-np.pi/2)]])
    coordinates=np.zeros(shape=(n_nodes,2))
    for i in range(n_nodes):
        n_row=int(i/N_2)
        n_col=i % N_2
        coordinates[i,:]=A.dot(np.array([n_row,n_col]-np.array([N_1/2,N_2/2])))
        
    coordinates[:,0]/=np.max(np.abs(coordinates[:,0]))
    coordinates[:,1]/=np.max(np.abs(coordinates[:,1]))
    coordinates*=2.0

    data_ref=[np.random.multivariate_normal(mean=np.array([0,0]),cov=np.eye(2),size=N_ref) for i in range(n_nodes)]
   # data_test=[np.random.multivariate_normal(mean=coordinates[i,:],cov=np.eye(2),size=N_test) for i in range(n_nodes)]
    
    data_test=[np.random.multivariate_normal(mean=coordinates[i,:],cov=np.diag(1.0+np.abs(coordinates[i,:])),size=N_test) for i in range(n_nodes)]

    return G,data_ref,data_test,affected_nodes


def generate_experiment_4_LRE(n_nodes,N_ref,N_test,d=50):
    
    ######### This function implements experiments 2B and 2C from the paper.

    ####### Input
    # n_nodes: number of nodes in the graph.
    # N_ref: number of observations from p_v.
    # N_test: number of observations from q_v.
    # d: the final dimension of the input space is 2 * d.

    ###### Output 
    # G: the graph component of the problem.
    # data_ref: a list of n_nodes elements, each representing observations coming from p_v.
    # data_test: a list of n_nodes elements, each representing observations coming from q_v.
    # affected_nodes: a numpy array of the nodes affected during the experiment.

    
    N_1=int(np.sqrt(n_nodes))
    N_2=int(np.ceil(n_nodes/N_1))
    
    G=graphs.Grid2d(N1=N_1,N2=N_2)
    n_nodes=int(N_1*N_2)
    
    affected_nodes=[]
    
    for i in range(int(N_1/2)):
        affected_nodes.append(N_2*i+np.arange(int(N_2)))
        
    for i in range(int(N_1/2),N_1):
        affected_nodes.append(N_2*i+np.arange(int((N_2+1)/2)))
      
    affected_nodes=np.hstack(affected_nodes)
    

    data_ref=[np.random.multivariate_normal(mean=np.zeros(int(d*2)),cov=np.eye(int(d*2)),size=N_ref) for i in range(n_nodes)]
    data_test=[np.random.multivariate_normal(mean=np.zeros(int(d*2)),cov=np.eye(int(d*2)),size=N_test) for i in range(n_nodes)]
    
    cov_blocks=np.eye(int(d*2))
    for i in range(int(d)):
        cov_blocks[2*i,2*i+1]=0.8
        cov_blocks[2*i+1,2*i]=0.8

    for i in affected_nodes:
        data_test[i]=np.random.multivariate_normal(mean=np.zeros(int(d*2)),cov=cov_blocks,size=N_test)

    return G,data_ref,data_test,affected_nodes

#### Real likelihood ratio. This set of functions is later used to estimate the errors in likelihood-ratio estimation.


def real_relative_likelihood_ratio_highdimension(x,cov_blocks,d=50,alpha=0.1):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(int(d*2)),cov=np.eye(int(d*2)))+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(int(d*2)),cov=cov_blocks)
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(int(d*2)),cov=cov_blocks)
    return np.exp(log_r_alpha)

def real_relative_likelihood_ratio_grid(x,coordinates,alpha=0.1):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),cov=np.eye(2))+alpha*scipy.stats.multivariate_normal.pdf(x,coordinates,cov=np.diag(1.0+np.abs(coordinates)))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,coordinates,cov=np.diag(1.0+np.abs(coordinates)))
    return np.exp(log_r_alpha)

def real_relative_likelihood_ratio_C_1(x,alpha=0.1):
    return uniform.pdf(x,loc=-np.sqrt(3),scale=2*np.sqrt(3))/(alpha*uniform.pdf(x,loc=-np.sqrt(3),scale=2*np.sqrt(3))+(1-alpha)*norm.pdf(x))


def real_relative_likelihood_ratio_C_2(x,alpha=0.1):
    return norm.pdf(x,loc=1)/(alpha*norm.pdf(x,loc=1)+(1-alpha)*norm.pdf(x))


def real_relative_likelihood_ratio_C_3_C_4(x,alpha=0.1):
    return norm.pdf(x)/(alpha*norm.pdf(x)+(1-alpha)*norm.pdf(x))


def r_mul_normal_C1(x,alpha):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))
    return np.exp(log_r_alpha)

def r_mul_normal_C2(x,alpha):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(2),np.array([[1,-4/5],[-4/5,1]]))
    return np.exp(log_r_alpha)

def r_mul_normal_C3(x,alpha):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,4/5],[4/5,1]]))+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,0],[0,1]]))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(2),np.array([[1,0],[0,1]]))
    return np.exp(log_r_alpha)

def r_mul_normal_C4(x,alpha):
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,0],[0,1]]))+alpha*scipy.stats.multivariate_normal.pdf(x,np.ones(2),np.array([[1,0],[0,1]]))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.ones(2),np.array([[1,0],[0,1]]))
    return np.exp(log_r_alpha)


def r_mul_normal_non_C(x,alpha):
    sigma=np.eye(3)
    sigma[1,2]=4/5
    sigma[2,1]=4/5
    mu_2=np.array((0,0,0))
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(3),sigma)+alpha*scipy.stats.multivariate_normal.pdf(x,mu_2,sigma)
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(3),sigma)
    return np.exp(log_r_alpha)

def r_mul_normal_non_C_2B(x,alpha):
    sigma=10*np.eye(2)
    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),sigma)+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(2),sigma)
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(2),sigma)
    return np.exp(log_r_alpha)


def r_mul_normal_C(x,alpha):
    sigma_1=np.eye(3)
    sigma_1[1,2]=4/5
    sigma_1[2,1]=4/5
    
    sigma_2=np.eye(3)

    log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,np.zeros(3),sigma_1)+alpha*scipy.stats.multivariate_normal.pdf(x,np.zeros(3),sigma_2)
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=scipy.stats.multivariate_normal.logpdf(x,np.zeros(3),sigma_2)
    return np.exp(log_r_alpha)

def r_normal_mixture(x,alpha):
    means=[np.array([0,0]),np.array([0,5]),np.array([0,-5]),np.array([5,0]),np.array([-5,0])]   
    log_r_alpha=[scipy.stats.multivariate_normal.logpdf(x,mean=m,cov=5*np.eye(2)) for m in means]
    log_r_alpha=np.vstack(log_r_alpha)
    log_r_alpha=logsumexp(log_r_alpha,axis=0)
    aux_log_r_alpha=(1-alpha)*scipy.stats.multivariate_normal.pdf(x,mean=np.array([0,0]),cov=10*np.eye(2))
    aux_log_r_alpha+=alpha*np.exp(log_r_alpha)*(1/5)
    aux_log_r_alpha=np.log(aux_log_r_alpha)
    log_r_alpha=log_r_alpha-aux_log_r_alpha
    r_alpha=np.exp(log_r_alpha)*(1/5)
    return r_alpha

















