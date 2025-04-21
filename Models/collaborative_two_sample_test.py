# -----------------------------------------------------------------------------------------------------------------
# Title:  collaborative_two_sample_test.py
# Author(s): Alejandro de la Concha
# Initial version:  2020-05-17
# Last modified:    2025-02-27
# This version:     2025-02-27
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This script implements the two-sample test presented in the paper 
#               "Collaborative Non-Parametric Two-Sample Testing."
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy, scipy, joblib, dill
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GRULSIF, f-divergence, hypothesis testing
# -----------------------------------------------------------------------------------------------------------------


from Models.aux_functions import *
from Models.likelihood_ratio_collaborative import *
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import dill as pickle


def run_permutation_GRULSIF(permutation, W, cols, row_pointer, phi, alpha, gamma, lamb, N_ref, N_test, degrees, tol):
    ###### This function runs GRULSIF for a given permutation.

    ### Input
    # permutation: the permutation of the indices used to run the method.
    # W: graph structure in CSR format.
    # cols: the column index where the observation is different from 0.
    # row_pointer: the row index corresponding to the column that is different from zero.
    # phi: precomputed matrix of feature maps to reduce computational cost.
    # alpha: regularization parameter associated with the upperbound of the relative likelihood ratio it should be between (0,1)
    # gamma: regularization constant associated with the norm in H.
    # lamb: regularization constant associated with the graph component.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.
    # degrees: the degree associated with each of the nodes.
    # tol: the accepted tolerance in the estimation.

    ### Output
    # score: the Pearson divergence associated with each of the nodes.
    
    n_nodes = len(phi)
    L = len(phi[0][0])
    h_test = np.zeros((n_nodes, L), dtype=np.float32)
    A = np.zeros((n_nodes, L, L), dtype=np.float32)
    learning_rates = np.zeros(n_nodes)

    for i in range(n_nodes):
        h_test[i] = np.sum(phi[i][permutation][N_ref:], axis=0)
        h_test[i] /= N_test
        A[i] = np.einsum('ji,j...', phi[i][permutation]
                         [:N_ref], phi[i][permutation][:N_ref])
        A[i] *= (1-alpha)/N_ref
        A[i] += np.einsum('ji,j...', phi[i][permutation][N_ref:],
                          phi[i][permutation][N_ref:])*(alpha/N_test)
        A[i] /= n_nodes
        A[i] += lamb*degrees[i]*np.eye(L)
        eta_i, _ = eigsh(A[i], k=1)
        learning_rates[i] = 1.*eta_i

    theta_ini = 1e-6*np.ones((n_nodes, L), dtype=np.float32)

    theta = optimize_GRULSIF(theta_ini, W=W, cols=cols, row_pointer=row_pointer, A=A, h_test=h_test, learning_rates=learning_rates, alpha=alpha, gamma=gamma,
                             lamb=lamb, tol=tol, verbose=False)

    for i in range(n_nodes):
        A[i] -= lamb*degrees[i]*np.eye(L)
        A[i] *= n_nodes

    score = np.zeros(n_nodes)
    for i in range(n_nodes):
        score[i] = theta[i].dot(A[i]).dot(theta[i])
        score[i] *= -0.5
        score[i] += theta[i].dot(h_test[i])
        score[i] -= 0.5

    return score


def run_permutation_Pool(permutation, phi, alpha, gamma, N_ref, N_test, tol):
    ###### This function runs Pool for a given permutation 
    
    ### Input
    # permutation: the permutation of the indices used to run the method.
    # phi: precomputed matrix of feature maps to reduce computational cost.
    # alpha: regularization parameter associated with the upperbound of the relative likelihood ratio it should be between (0,1)
    # gamma: regularization constant associated with the norm in H.
    # N_ref: number of observations coming from p_v.
    # N_test: number of observations coming from q_v.
    # tol: the accepted tolerance in the estimation.

    ### Output
    # score: the Pearson divergence associated with each of the nodes.
    
    n_nodes = len(phi)
    L = len(phi[0][0])
    h_test = np.zeros((n_nodes, L), dtype=np.float32)
    A = np.zeros((n_nodes, L, L), dtype=np.float32)

    for i in range(n_nodes):
        h_test[i] = np.sum(phi[i][permutation][N_ref:], axis=0)
        h_test[i] /= N_test
        A[i] = np.einsum('ji,j...', phi[i][permutation]
                         [:N_ref], phi[i][permutation][:N_ref])
        A[i] *= (1-alpha)/N_ref
        A[i] += np.einsum('ji,j...', phi[i][permutation][N_ref:],
                          phi[i][permutation][N_ref:])*(alpha/N_test)

    theta_ini = 1e-6*np.ones((n_nodes, L), dtype=np.float32)
    learning_rates = np.zeros(n_nodes)

    for i in range(n_nodes):
        A[i] /= n_nodes
        A[i] += 1e-6*np.eye(L)
        eta_i, _ = eigsh(A[i], k=1, ncv=np.min((L, 100)))
        learning_rates[i] = 1.*eta_i

    theta = optimize_Pool(theta_ini, A=A, learning_rates=learning_rates,
                          h_test=h_test, alpha=alpha, gamma=gamma, tol=tol, verbose=False)

    for i in range(n_nodes):
        A[i] -= 1e-6*np.eye(L)
        A[i] *= n_nodes

    score = np.zeros(n_nodes)
    for i in range(n_nodes):
        score[i] = theta[i].dot(A[i]).dot(theta[i])
        score[i] *= -0.5
        score[i] += theta[i].dot(h_test[i])
        score[i] -= 0.5

    return score



class C2ST():
    # This class implements the Collaborative Two-Sample Test.

    def __init__(self, W, data_ref, data_test, threshold_coherence=0.3, alpha=0.1, tol=1e-3, k_cross_validation=5, verbose=False, time=False):
        ### Input
        # W: the adjacency matrix in CSR format (if the test includes a time component, a multiplex is built instead).
        # data_ref: data points representing the distribution p_v(.).
        # data_test: data points representing the distribution q_v(.).
        # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
        #                      When the kernel is normal, this parameter should be between 0 and 1.
        #                      The closer it is to 1, the larger the dictionary and the slower the training.
        # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
        # tol: accepted tolerance level in the estimation.
        # k_cross_validation: number of splits for cross-validation.
        # verbose: whether or not to plot the model selection results.
        # time: whether or not the time component is considered in the test.
        
        self.n_nodes=W.shape[0]
        
        try: 
            n_nodes_1=len(data_ref)
            n_nodes_2=len(data_test)
            if not (n_nodes_1==self.n_nodes and n_nodes_1==self.n_nodes):
                raise ValueError(F"The datasets should be list with as many elements as the numbers of nodes")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
            
        
        try:
            alpha=float(alpha)
            if not (0.0<=alpha<1):
                raise ValueError(F"Parameter alpha must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)    
        except TypeError as e:
            print("Error: alpha parameter should be a float")
            sys.exit(1)
            
        try:
            k_cross_validation=int(k_cross_validation)
            if not (1<k_cross_validation):
                raise ValueError(F"The number of samples should be bigger than 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
        try:
            threshold_coherence=float(threshold_coherence)
            if not (0.0<threshold_coherence<1):
                raise ValueError(F"The threshold coherence must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: The threshold coherence parameter should be a float")
            sys.exit(1)
            
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except TypeError as e:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)
            
            
        try: 
            Ws=(W+W.T)/2
            if not (abs(Ws-Ws.T)>1e-10).nnz == 0: 
                raise SymmetryError(F"The weight matrix should be symmetric")
            if not (W<0).nnz==0:
                raise ValueError(F"All the elements of the weight matrix should bigger or equal to zero ")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except SymmetryError as e:
            print(f"Error: {e}")
            sys.exit(1)


        self.time = time

        if time:
            self.n_times = data_ref[0].shape[0]
            self.N_ref = data_ref[0].shape[1]
            self.N_test = data_test[0].shape[1]
            self.n_nodes = len(data_ref)
            self.W = transform_matrix_totime(W.tocoo(), self.n_times)
            self.W = self.W.tocoo()
            data_ref = [transform_data(data_ref[node][t]) for t, node in list(
                product(range(self.n_times), range(self.n_nodes)))]
            data_test = [transform_data(data_test[node][t]) for t, node in list(
                product(range(self.n_times), range(self.n_nodes)))]
            self.n_nodes = len(data_ref)

        else:
            self.W = W.tocoo()
            self.N_ref = len(data_ref[0])
            self.N_test = len(data_test[0])
            self.n_nodes = len(data_ref)
            data_ref = [transform_data(d) for d in data_ref]
            data_test = [transform_data(d) for d in data_test]

        self.alpha = alpha
        self.tol = tol

        self.W_ij = self.W.data
        row_indices = self.W.row
        self.col_indices = self.W.col

        sort_order = np.argsort(row_indices)
        self.W_ij = self.W_ij[sort_order]
        self.col_indices = self.col_indices[sort_order]

        num_rows = len(data_ref)
        self.row_ptr = np.empty(num_rows + 1, dtype=int)
        current_row = 0
        for i in range(len(row_indices)):
            while current_row < row_indices[i]:
                self.row_ptr[current_row + 1] = i
                current_row += 1
        self.row_ptr[-1] = len(row_indices)

        self.grulsif_1 = GRULSIF(self.W, data_ref, data_test, threshold_coherence,
                                 alpha, tol, k_cross_validation, verbose)
        self.grulsif_2 = GRULSIF(self.W, data_test, data_ref, threshold_coherence,
                                 alpha, tol, k_cross_validation, verbose)
        phi_ref_1 = [self.grulsif_1.kernel.k_V(d) for d in data_ref]
        phi_test_1 = [self.grulsif_1.kernel.k_V(d) for d in data_test]
        phi_ref_2 = [self.grulsif_2.kernel.k_V(d) for d in data_ref]
        phi_test_2 = [self.grulsif_2.kernel.k_V(d) for d in data_test]
        self.theta_1, self.score_pq = self.fit(
            phi_ref_1, phi_test_1, self.grulsif_1)
        self.theta_2, self.score_qp = self.fit(
            phi_test_2, phi_ref_2, self.grulsif_2)

        self.concatenate_data_1 = [
            np.vstack((d_r, d_t)) for d_r, d_t in zip(phi_ref_1, phi_test_1)]
        self.concatenate_data_2 = [
            np.vstack((d_r, d_t)) for d_r, d_t in zip(phi_ref_2, phi_test_2)]

    def fit(self, phi_ref, phi_test, LRE_model):
       ##### This function returns the theta parameters related to the likelihood ratios 
       ##### and the Pearson divergence at each node, computed by integrating the feature maps phi_ref and phi_test.

        ### Input
        # phi_ref: the feature map over the set X.
        # phi_test: the feature map over the set X'.
        # LRE_model: the likelihood-ratio model used for estimation.

        ### Output
        # theta: a numpy matrix of dimension n_nodes x L, where L is the size of the dictionary. 
        #        Each row represents a node in the graph.
        # score: a vector of dimension n_nodes, where each entry is the Pearson divergence estimate associated with that node.

        
        n_nodes = len(phi_ref)
        L = phi_ref[0].shape[1]
        h_test = np.zeros((n_nodes, L), dtype=np.float32)
        A = np.zeros((n_nodes, L, L), dtype=np.float32)
        for i in range(n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i] /= self.N_test
            A[i] = np.einsum('ji,j...',  phi_ref[i], phi_ref[i])
            A[i] *= (1-self.alpha)/self.N_ref
            A[i] += np.einsum('ji,j...',  phi_test[i],
                              phi_test[i])*(self.alpha/self.N_test)

        theta_ini = 1e-6*np.ones((n_nodes, L), dtype=np.float32)
        learning_rates = np.zeros(n_nodes)
        for i in range(n_nodes):
            A[i] /= n_nodes
            A[i] += LRE_model.lamb*LRE_model.degrees[i]*np.eye(L)
            eta_i, _ = eigsh(A[i], k=1, ncv=np.min((L, 100)))
            learning_rates[i] = 1.*eta_i

        theta = optimize_GRULSIF(theta_ini, W=self.W_ij, cols=self.col_indices, row_pointer=self.row_ptr,
                                 A=A, h_test=h_test, learning_rates=learning_rates, alpha=self.alpha, gamma=LRE_model.gamma,
                                 lamb=LRE_model.lamb, tol=self.tol, verbose=False)

        for i in range(n_nodes):
            A[i] -= LRE_model.lamb*LRE_model.degrees[i]*np.eye(L)
            A[i] *= n_nodes

        score = np.zeros(n_nodes)
        for i in range(n_nodes):
            score[i] = theta[i].dot(A[i]).dot(theta[i])
            score[i] *= -0.5
            score[i] += theta[i].dot(h_test[i])
            score[i] -= 0.5

        return theta, score

    def aux_get_divergences(self, permutations):
        ### This function computes the Pearson divergence associated with a set of permutations.
        ### Input:
        # permutations: a list of permutations.
        ### Output:
            # PE_1, PE_2: lists of Pearson divergence scores, depending on the order in which the datasets are taken.

        n_permutations = permutations.shape[0]
        n_nodes = len(self.concatenate_data_1)
        PE_1 = np.zeros((n_permutations, n_nodes))
        PE_2 = np.zeros((n_permutations, n_nodes))

        for i in range(n_permutations):
            hat_phi_ref = [d[permutations[i]][:self.N_ref]
                           for d in self.concatenate_data_1]
            hat_phi_test = [d[permutations[i]][self.N_ref:]
                            for d in self.concatenate_data_1]
            _, PE_1[i] = self.fit(hat_phi_ref, hat_phi_test, self.grulsif_1)
            hat_phi_ref = [d[permutations[i]][:self.N_ref]
                           for d in self.concatenate_data_2]
            hat_phi_test = [d[permutations[i]][self.N_ref:]
                            for d in self.concatenate_data_2]
            _, PE_2[i] = self.fit(hat_phi_test, hat_phi_ref, self.grulsif_2)

        return PE_1, PE_2

    def run_permutations(self, n_permutations):
        #### Function generating n_permutations over the index set {0, 1, ..., N_ref + N_test}.
        ### Input:
        # n_permutations: the number of permutations to generate.
        ### Output:
        # permutations: a list of n_permutations randomly shuffled index sequences.

        permutations = np.zeros((n_permutations, self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i] = np.random.permutation(self.N_ref+self.N_test)
        permutations = permutations.astype(int)

        return permutations

    def get_pivalues(self, n_rounds=1000):
        #### The p-values are computed via a permutation test.

        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.

        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.

        
        permutations = self.run_permutations(n_rounds)
        PD_1, PD_2 = self.aux_get_divergences(permutations)

        self.PD_1s = PD_1
        self.PD_2s = PD_2

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


    def get_pivalues_multiprocessing(self, n_rounds=1000):
        #### The p-values are computed via a permutation test. When the equipment allows it, multiprocessing is used to speed up computations.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.

        
        permutations = self.run_permutations(n_rounds)

        PD_1 = Parallel(n_jobs=-1, prefer="threads")(delayed(run_permutation_GRULSIF)(p, self.W_ij, self.col_indices, self.row_ptr,
                                                                                      self.concatenate_data_1, self.alpha, self.grulsif_1.gamma,
                                                                                      self.grulsif_1.lamb, self.N_ref, self.N_test, self.grulsif_1.degrees,
                                                                                      tol=self.tol) for p in permutations)

        PD_2 = Parallel(n_jobs=-1, prefer="threads")(delayed(run_permutation_GRULSIF)(p, self.W_ij, self.col_indices, self.row_ptr,
                                                                                      self.concatenate_data_2, self.alpha, self.grulsif_2.gamma,
                                                                                      self.grulsif_2.lamb, self.N_ref, self.N_test, self.grulsif_2.degrees,
                                                                                      tol=self.tol) for p in permutations)

        self.PD_1s = PD_1
        self.PD_2s = PD_2

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



class Pool_two_sample_test():
    # This class implements the two sampling test at the node level via a permutation test ignoring the graph structure
    def __init__(self, data_ref, data_test, threshold_coherence=0.3, alpha=0.1, tol=1e-3, k_cross_validation=5, verbose=False, time=False):
        ### Input
        # data_ref: data points representing the distribution p_v(.).
        # data_test: data points representing the distribution q_v(.).
        # threshold_coherence: parameter related to dictionary selection, as described in Richard et al. (2009).
        #                      When the kernel is normal, this parameter should be between 0 and 1.
        #                      The closer it is to 1, the larger the dictionary and the slower the training.
        # alpha: regularization parameter associated with the upper bound of the likelihood ratio.
        # tol: accepted tolerance level in the estimation.
        # k_cross_validation: number of splits for cross-validation.
        # verbose: whether or not to plot the model selection results.
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
            
        try:
            k_cross_validation=int(k_cross_validation)
            if not (1<k_cross_validation):
                raise ValueError(F"The number of splits should be bigger than 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  # Stops execution
            
        try:
            threshold_coherence=float(threshold_coherence)
            if not (0.0<threshold_coherence<1):
                raise ValueError(F"The threshold coherence must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError as e:
            print("Error: The threshold coherence parameter should be a float")
            sys.exit(1)  
            
        try:
            tol=float(tol)
            if not (0.0<tol<1):
                raise ValueError(F"The convergence tolerance must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)  
        except TypeError:
            print("Error: The convergence tolerance parameter should be a float")
            sys.exit(1)  
            
        self.time = time

        if time:
            self.n_times = data_ref[0].shape[0]
            self.N_ref = data_ref[0].shape[1]
            self.N_test = data_test[0].shape[1]
            self.n_nodes = len(data_ref)
            data_ref = [transform_data(data_ref[node][t]) for t, node in list(
                product(range(self.n_times), range(self.n_nodes)))]
            data_test = [transform_data(data_test[node][t]) for t, node in list(
                product(range(self.n_times), range(self.n_nodes)))]
            self.n_nodes = len(data_ref)

        else:
            self.N_ref = len(data_ref[0])
            self.N_test = len(data_test[0])
            self.n_nodes = len(data_ref)
            data_ref = [transform_data(d) for d in data_ref]
            data_test = [transform_data(d) for d in data_test]

        self.alpha = alpha
        self.tol = tol

        self.pool_1 = Pool(data_ref, data_test, threshold_coherence,
                           alpha, tol, k_cross_validation, verbose)
        self.pool_2 = Pool(data_test, data_ref, threshold_coherence,
                           alpha, tol, k_cross_validation, verbose)

        phi_ref_1 = [self.pool_1.kernel.k_V(d) for d in data_ref]
        phi_test_1 = [self.pool_1.kernel.k_V(d) for d in data_test]
        phi_ref_2 = [self.pool_2.kernel.k_V(d) for d in data_ref]
        phi_test_2 = [self.pool_2.kernel.k_V(d) for d in data_test]
        self.theta_1, self.score_pq = self.fit(
            phi_ref_1, phi_test_1, self.pool_1)
        self.theta_2, self.score_qp = self.fit(
            phi_test_2, phi_ref_2, self.pool_2)

        self.concatenate_data_1 = [
            np.vstack((d_r, d_t)) for d_r, d_t in zip(phi_ref_1, phi_test_1)]
        self.concatenate_data_2 = [
            np.vstack((d_r, d_t)) for d_r, d_t in zip(phi_ref_2, phi_test_2)]

    def fit(self, phi_ref, phi_test, LRE_model):
       ##### This function returns the theta parameters related to the likelihood ratios 
       ##### and the Pearson divergence at each node, computed by integrating the feature maps phi_ref and phi_test.

        ### Input
        # phi_ref: the feature map over the set X.
        # phi_test: the feature map over the set X'.
        # LRE_model: the likelihood-ratio model used for estimation.

        ### Output
        # theta: a numpy matrix of dimension n_nodes x L, where L is the size of the dictionary. 
        #        Each row represents a node in the graph.
        # score: a vector of dimension n_nodes, where each entry is the Pearson divergence estimate associated with that node.

        
        L = phi_ref[0].shape[1]
        n_nodes = len(phi_ref)
        h_test = np.zeros((n_nodes, L), dtype=np.float32)
        A = np.zeros((n_nodes, L, L), dtype=np.float32)
        for i in range(n_nodes):
            h_test[i] = np.sum(phi_test[i], axis=0)
            h_test[i] /= self.N_test
            A[i] = np.einsum('ji,j...',  phi_ref[i], phi_ref[i])
            A[i] *= (1-self.alpha)/self.N_ref
            A[i] += np.einsum('ji,j...',  phi_test[i],
                              phi_test[i])*(self.alpha/self.N_test)

        theta_ini = 1e-6*np.ones((n_nodes, L), dtype=np.float32)

        learning_rates = np.zeros(n_nodes)
        for i in range(n_nodes):
            A[i] /= n_nodes
            A[i] += 1e-6*np.eye(L)
            eta_i, _ = eigsh(A[i], k=1, ncv=np.min((L, 500)))
            learning_rates[i] = 1*eta_i

        theta = optimize_Pool(theta_ini, A, h_test, learning_rates,
                              self.alpha, LRE_model.gamma, tol=self.tol, verbose=False)

        for i in range(n_nodes):
            A[i] -= 1e-6*np.eye(L)
            A[i] *= n_nodes

        score = np.zeros(n_nodes)

        for i in range(n_nodes):
            score[i] = theta[i].dot(A[i]).dot(theta[i])
            score[i] *= -0.5
            score[i] += theta[i].dot(h_test[i])
            score[i] -= 0.5

        return theta, score

    def aux_get_divergences(self, permutations):
        ### This function computes the Pearson divergence associated with a set of permutations.
        ### Input:
        # permutations: a list of permutations.
        ### Output:
            # PE_1, PE_2: lists of Pearson divergence scores, depending on the order in which the datasets are taken.

        n_nodes = len(self.concatenate_data_1)
        n_permutations = permutations.shape[0]
        PE_1 = np.zeros((n_permutations, n_nodes))
        PE_2 = np.zeros((n_permutations, n_nodes))

        for i in range(n_permutations):
            hat_phi_ref = [d[permutations[i]][:self.N_ref]
                           for d in self.concatenate_data_1]
            hat_phi_test = [d[permutations[i]][self.N_ref:]
                            for d in self.concatenate_data_1]
            _, PE_1[i] = self.fit(hat_phi_ref, hat_phi_test, self.pool_1)
            hat_phi_ref = [d[permutations[i]][:self.N_ref]
                           for d in self.concatenate_data_2]
            hat_phi_test = [d[permutations[i]][self.N_ref:]
                            for d in self.concatenate_data_2]
            _, PE_2[i] = self.fit(hat_phi_test, hat_phi_ref, self.pool_2)

        return PE_1, PE_2

    def run_permutations(self, n_permutations):
        #### Function generating n_permutations over the index set {0, 1, ..., N_ref + N_test}.
        ### Input:
        # n_permutations: the number of permutations to generate.
        ### Output:
        # permutations: a list of n_permutations randomly shuffled index sequences.

        permutations = np.zeros((n_permutations, self.N_ref+self.N_test))
        for i in range(n_permutations):
            permutations[i] = np.random.permutation(self.N_ref+self.N_test)
        permutations = permutations.astype(int)

        return permutations

    def get_pivalues(self,  n_rounds=1000):
        #### The p-values are computed via permutation test 
        # Input
        # n_rounds: Number of permutations used to estimated the p-values
        # Output
        # p-values_1,p_values_2: p_values associated to each side of the PEARSON divergence estimates.
        
        
        permutations = self.run_permutations(n_rounds)
        PD_1, PD_2 = self.aux_get_divergences(permutations)
        
        self.PD_1s = PD_1
        self.PD_2s = PD_2

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

    def get_pivalues_multiprocessing(self, n_rounds=1000):
        #### The p-values are computed via a permutation test. When the equipment allows it, multiprocessing is used to speed up computations.
        ### Input:
        # n_rounds: the number of permutations used to estimate the p-values.
        ### Output:
        # p_values_1, p_values_2: p-values associated with each side of the Pearson divergence estimates.
        
        permutations = self.run_permutations(n_rounds)

        PD_1 = Parallel(n_jobs=-1, prefer="threads")(delayed(run_permutation_Pool)(p, self.concatenate_data_1, self.alpha, self.pool_1.gamma,
                                                                                   self.N_ref, self.N_test, tol=self.tol) for p in permutations)

        PD_2 = Parallel(n_jobs=-1, prefer="threads")(delayed(run_permutation_Pool)(p, self.concatenate_data_2, self.alpha, self.pool_2.gamma,
                                                                                   self.N_ref, self.N_test, tol=self.tol) for p in permutations)

        self.PD_1s = PD_1
        self.PD_2s = PD_2

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

