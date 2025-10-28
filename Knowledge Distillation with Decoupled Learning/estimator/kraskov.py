from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import numpy as np

def kraskov_estimator(X, Y, k=3):
    """
    Kraskov estimator of mutual information between two continuous multivariate variables X and Y.
    
    Args:
        X (np.ndarray): shape (n_samples, n_features1)
        Y (np.ndarray): shape (n_samples, n_features2)
        k (int): Number of neighbors to use
    
    Returns:
        mi (float): Estimated mutual information (MI)
    """
    assert len(X) == len(Y), "X and Y must have the same number of samples"
    n = len(X)
    
    # joint entropy to measure total uncertainty
    Z = np.concatenate((X, Y), axis=1)
    
    # fit k-NN in joint space
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')  
    nbrs.fit(Z)
    distances, _ = nbrs.kneighbors(Z)
    eps = distances[:, k]  # distance to k-th neighbor

    # count neighbors in marginal spaces within eps radius
    nx = []
    ny = []

    nbrs_x = NearestNeighbors(radius=1e10, metric='chebyshev').fit(X)
    nbrs_y = NearestNeighbors(radius=1e10, metric='chebyshev').fit(Y)
    
    for i in range(n):
        nx_i = nbrs_x.radius_neighbors([X[i]], radius=eps[i] - 1e-10, return_distance=False)[0].shape[0] - 1
        ny_i = nbrs_y.radius_neighbors([Y[i]], radius=eps[i] - 1e-10, return_distance=False)[0].shape[0] - 1
        nx.append(nx_i)
        ny.append(ny_i)

    nx = np.array(nx)
    ny = np.array(ny)

    # Kraskov formula see ref.
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))

    return mi
