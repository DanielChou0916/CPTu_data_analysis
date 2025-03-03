import numpy as np
from scipy import stats
from itertools import product

def gauss_legendre(n, a, b):
    """
    Calculate n-point Gauss-Legendre quadrature within [a, b]

    Parameters:
    n(int): Number of quadrature points.
    a(float): Lower bound of integration.
    b(float): Upper bound of integration.

    Returns:
    p(ndarray): Quadrature points.
    w(ndarray): Quadrature weights.
    """
    m = int(np.ceil(n / 2.0))
    x = np.zeros(n)
    w = np.zeros(n)

    for ii in range(1, m + 1):
        z = np.cos(np.pi * (ii - 0.25) / (n + 0.5))
        z1 = z + 1
        while abs(z - z1) > 1e-10:
            p1 = 1.0
            p2 = 0.0
            for jj in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * jj - 1) * z * p2 - (jj - 1) * p3) / jj
            pp = n * (z * p1 - p2) / (z**2 - 1)
            z1 = z
            z = z1 - p1 / pp
        
        x[ii - 1] = -z
        x[-ii] = z
        w[ii - 1] = 2 / ((1 - z**2) * pp**2)
        w[-ii] = w[ii - 1]

    # Scale the nodes and weights to the interval [a, b]
    p = (b - a) / 2 * x + (b + a) / 2
    w = (b - a) / 2 * w

    return p, w

def GLJointNW(M, N, a, b, c, d):
    """
    Gauss-Legendre method to generate nodes and weights for double
    integration.
    
    Parameters:
    - M, N: Number of points for the quadrature in each dimension.
    - a, b: Integration limits for the first variable.
    - c, d: Integration limits for the second variable.
    
    Returns:
    - A numpy array of shape (M * N, 3) with:
      - The first column being the variable x_p within integration range {a, b}.
      - The second column being the variable x_q within integration range {c, d}.
      - The third column being the joint weights W_pq accounting for the integral ranges.
    """
    # gauss_legendre is defined elsewhere to return nodes and weights
    x_p, w_p = gauss_legendre(M, a, b)
    y_q, w_q = gauss_legendre(N, c, d)
    
    # Initialize the output array
    Gauss_points_weights = np.zeros((M * N, 3))
    
    # Fill in the output array
    counter = 0
    for i in range(M):
        for j in range(N):
            Gauss_points_weights[counter] = [x_p[i], y_q[j], w_p[i] * w_q[j]]
            counter += 1
    
    return Gauss_points_weights

def GLJointNW_general(N, Lower, Upper):
    """
    General Gauss-Legendre method to generate nodes and weights for multiple integration,
    applicable in numerical integration tasks across multiple dimensions.
    
    Parameters:
    - N (list of int): Number of Gauss-Legendre quadrature points for each dimension.
    - Lower (list of float): Lower bounds of integration for each dimension.
    - Upper (list of float): Upper bounds of integration for each dimension.
    
    Returns:
    1. Gauss_points_weights (numpy array): A multi-dimensional grid for numerical integration.
       - Shape: (prod(N), len(N)+1). 
       - Contains variables x_i within [Lower[i], Upper[i]] for each ith dimension and the last
         column contains the joint weights W_p... for integral calculations.
    2. GLpw (list of tuples): Contains nodes (Gauss points) and weights for each dimension,
       useful for separate or joint variable integration.
       - Each tuple: (Gauss points, weights) for individual dimension integration.
       
    Example:
    To integrate a 2D function over [0,1]x[0,1] with 10 points in each dimension:
        N, Lower, Upper = [10, 10], [0, 0], [1, 1]
        points_weights, individual_components = GLJointNW_general(N, Lower, Upper)
    """
    GLpw = []
    for node_num, lower, upper in zip(N, Lower, Upper):
        x_p, w_p = gauss_legendre(node_num, lower, upper)
        GLpw.append((x_p, w_p))
    
    # Initialize the output array
    num_dims = len(N)
    total_points = np.prod(N)
    Gauss_points_weights = np.zeros((total_points, num_dims + 1))
    
    # Generate all combinations of indices for the nodes
    indices = [range(len(x_p[0])) for x_p in GLpw]
    
    # Iterate over all combinations of nodes
    for idx, index_combination in enumerate(product(*indices)):
        # Compute the product of weights for the current combination
        weight_product = np.prod([GLpw[dim][1][i] for dim, i in enumerate(index_combination)])
        
        # Store the nodes for each dimension and the weight product
        node_values = [GLpw[dim][0][i] for dim, i in enumerate(index_combination)]
        Gauss_points_weights[idx, :-1] = node_values
        Gauss_points_weights[idx, -1] = weight_product
    
    return Gauss_points_weights,GLpw

def GL_entropy(density,weight,e=1e-12):
    """
    Numerical estimation of entropy using Gaussian quadrature.

    Parameters:
    - density (numpy array): A 1D array of density values obtained by evaluating the KDE function
                             at Gauss-Legendre quadrature points. These could represent the density
                             of a single variable or joint densities of multiple variables.
    - weight (numpy array): A 1D array of Legendre weights corresponding to the quadrature points.
                            The size of this array must match that of the `density` array.
    - e (float): A small correction term added to density values before taking the logarithm to 
                 ensure numerical stability. This prevents undefined log2(0) calculations and is 
                 particularly useful when density values are very close to or are zero, which can occur 
                 with narrow KDE bandwidth or when estimating joint densities.

    Returns:
    - Entropy (float): The estimated differential entropy based on the provided densities and weights.
                       This can represent marginal entropy if the input densities are for a single variable,
                       or joint entropy if the densities are for joint distributions of multiple variables.

    Example:
        density = kde_function(gauss_points)  # kde_function is your KDE estimation function
        weight = gauss_legendre_weights
        entropy = GL_entropy(density, weight)
    """
    adjusted_density = np.maximum(density, e)
    return -(adjusted_density * np.log2(adjusted_density)*weight).sum()

def GL_KLD(kdex, kdey, p, w, e=1e-12):
    """
    Estimate the Kullback-Leibler Divergence (KLD) between two distributions
    using Gaussian quadrature and kernel density estimates (KDE).

    Parameters:
    - kdex (function): KDE for distribution P, should accept an array of points.
    - kdey (function): KDE for distribution Q, should accept an array of points.
    - p (numpy array): Points at which KDEs are evaluated, obtained from Gaussian quadrature.
    - w (numpy array): Weights associated with the quadrature points.
    - e (float): Small constant to avoid log2(0), ensuring numerical stability.

    Returns:
    - KLD (float): The estimated Kullback-Leibler Divergence from P to Q in bits.
    """
    px = kdex(p)
    adj_px = np.maximum(px, e)  # Adjust density from P to avoid log2(0)
    py = kdey(p)
    adj_py = np.maximum(py, e)  # Adjust density from Q to avoid log2(0)

    # Calculate the KLD
    return ((adj_px * (np.log2(adj_px) - np.log2(adj_py))) * w).sum()

def GL_MI(kdex, kdey, kdexy, Gauss_points_weights, e=1e-12):
    """
    Numerical estimation of mutual information (MI) between two variables using 
    Gaussian quadrature and kernel density estimation (KDE).
    
    Parameters:
    - kdex (function): KDE function for the first variable X. This function should
                       accept a numpy array of points and return the estimated 
                       density at those points.
    - kdey (function): KDE function for the second variable Y, similar to `kdex`.
    - kdexy (function): Joint KDE function for variables X and Y. This function 
                        should accept a 2D numpy array of points (each row is a point 
                        in the joint space of X and Y) and return the estimated joint 
                        density at those points.
    - Gauss_points_weights (numpy array): An array containing the Gauss-Legendre 
                                          quadrature points and weights. The last 
                                          column should contain the weights, and the 
                                          other columns the quadrature points for 
                                          each dimension.
    - e (float): A small positive value to ensure numerical stability by preventing
                 division by zero or log of zero in density estimates. `e` is added
                 to density values before taking the logarithm.
    
    Returns:
    - MI (float): The estimated mutual information between X and Y, based on the
                  provided KDE functions and quadrature points. Mutual information
                  quantifies the amount of information obtained about one variable
                  through observing the other.

    Example:
        # Assume `kdex`, `kdey`, `kdexy`, and `Gauss_points_weights` are defined
        mutual_information = GL_MI(kdex, kdey, kdexy, Gauss_points_weights)
    """
    pxy = kdexy(Gauss_points_weights[:, :-1].T)
    adj_pxy = np.maximum(pxy, e)

    px = kdex(Gauss_points_weights[:, 0])
    adj_px = np.maximum(px, e)
    py = kdey(Gauss_points_weights[:, 1])
    adj_py = np.maximum(py, e)
    w = Gauss_points_weights[:, -1]
    
    return ((adj_pxy * (np.log2(adj_pxy) - np.log2(adj_px) - np.log2(adj_py))) * w).sum()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#Entropy Approximation by Covariance
def H_approx(Z):
    if len(Z.shape)<2:
        Z=Z.reshape(-1,1)
    # k is the number of variables (dimensions) in Z
    k = Z.shape[-1]
    
    # Compute the covariance matrix of Z
    Sigma = np.cov(Z.T)
    
    # Calculate the determinant of the covariance matrix
    try:
        det = np.linalg.det(Sigma)
    except:
        det=Sigma
    
    # Calculate the entropy approximation using the determinant
    # The formula includes the 0.5 * k * log2(2 * pi * e) term to account for the (2 * pi * e)^k in the full formula
    return 0.5 * k * np.log2(2 * np.pi * np.e) + 0.5 * np.log2(det)

################################## Advanced Functions #################################

# Function for calculating joint Entropy
def Joint_Entropy(joint,buffer_coefficient=0.3,n=64):
    """
    Calculate the joint entropy of a given joint dataset.

    Parameters:
    - joint (numpy.ndarray): An NxD array, containing N data points each with D features.
    - buffer_coefficient (float): A coefficient to determine the buffer added to the range
                                  when calculating the joint entropy, to avoid edge effects
                                  in kernel density estimation.
    - n (int): The number of Gaussian quadrature points to use in the numerical integration.

    Returns:
    - jointh (float): The calculated joint entropy of the dataset.

    The function uses Gaussian kernel density estimation to estimate the probability density
    of the joint dataset and numerical integration to calculate the entropy.
    """
    _,D=joint.shape

    lower=joint.min(0)
    upper=joint.max(0)
    buffer=buffer_coefficient*(upper-lower)
    Gauss_points_weights,_=GLJointNW_general(N=[n for i in range(D)],Lower=lower-buffer, Upper=upper+buffer)
    jkde = stats.gaussian_kde(joint.T)

    pj = jkde(Gauss_points_weights[:,:-1].T)

    jointh=GL_entropy(pj,Gauss_points_weights[:,-1],e=1e-12)
    return jointh

def Marginal_Entropy(x,buffer_coefficient=0.3,n=64):
    """
    Calculate the marginal entropy of a single-variable dataset.

    Parameters:
    - x (numpy.ndarray): An N-element array, containing N data points of a single feature.
    - buffer_coefficient (float): A coefficient to determine the buffer added to the range
                                  when calculating the marginal entropy, to avoid edge effects
                                  in kernel density estimation.
    - n (int): The number of Gaussian quadrature points to use in the numerical integration.

    Returns:
    - hx (float): The calculated marginal entropy of the dataset.

    The function uses Gaussian kernel density estimation to estimate the probability density
    of the dataset and numerical integration to calculate the entropy.
    """
    lower=x.min()
    upper=x.max()
    buffer=buffer_coefficient*(upper-lower)
    p,w=gauss_legendre(n=n, a=lower-buffer, b=upper+buffer)
    kde=stats.gaussian_kde(x)
    px=kde(p)
    hx=GL_entropy(px,w,e=1e-12)

    return hx

def Mutual_Information(x,y,buffer_coefficient=0.3,n=64,numerical_integration=True):
    """
    Calculate the mutual information between two single-variable datasets by numerical integration.

    Parameters:
    - x, y (numpy.ndarray): Two N-element arrays, each containing N data points of a single feature.
    - buffer_coefficient (float): A coefficient to determine the buffer added to the ranges
                                  when calculating the mutual information, to avoid edge effects
                                  in kernel density estimation.
    - n (int): The number of Gaussian quadrature points to use in the numerical integration.
    - numerical_integration (bool): A flag to determine whether to calculate the mutual
                                    information using numerical integration (if True) or
                                    a direct entropy calculation method (if False).

    Returns:
    - I(x;y) (float): The calculated mutual information between the two datasets.

    The function uses Gaussian kernel density estimation to estimate the probability densities
    of each dataset and their joint dataset, followed by numerical integration to calculate
    the mutual information. If numerical_integration is False, the function uses a direct
    entropy calculation method instead.
    """
    joint=np.hstack([x.reshape(-1,1),y.reshape(-1,1)])

    lower=joint.min(0)
    #print(lower)
    upper=joint.max(0)
    #print(upper)
    buffer=buffer_coefficient*(upper-lower)
    #print(buffer)
    Gauss_points_weights,GLpw=GLJointNW_general(N=[n,n],Lower=lower-buffer, Upper=upper+buffer)
    # Assume x and y are now continuous random variables
    # We will use Gaussian KDEs to estimate their densities

    # Estimating the KDEs for x, y, and joint variables
    kdex = stats.gaussian_kde(x)
    kdey = stats.gaussian_kde(y)
    jkde = stats.gaussian_kde(joint.T)
    if numerical_integration:
        MI=GL_MI(kdex,kdey,jkde,Gauss_points_weights,e=1e-12)
    else:
        p1, w1 = GLpw[0]
        p2, w2 = GLpw[1]
        px = kdex(p1)
        py = kdey(p2)
        pxy = jkde(Gauss_points_weights[:,:-1].T)
        hx=GL_entropy(px,w1,e=1e-12)
        hy=GL_entropy(py,w2,e=1e-12)
        hxy=GL_entropy(pxy,Gauss_points_weights[:,-1],e=1e-12)
        MI=max(0, hx + hy - hxy)
    return MI