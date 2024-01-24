import numpy as np
from scipy.stats import norm

def IQR(data, tau=0.75):
    """
    Calculate the Interquartile Range (IQR) of a given data set.

    Args:
        data (array-like): The data set for which the IQR is to be calculated.
        tau (float, optional): The percentile to use for the IQR calculation. 
                               Defaults to 0.75.

    Returns:
        float: The calculated IQR of the data set.
    """
    Q1 = np.percentile(data, 100*(1-tau))
    Q3 = np.percentile(data, 100*tau)
    IQR = Q3 - Q1
    return IQR

def sg(ksi, tau=0.75, N=1000):
    """
    Calculate the scale factor for a given probability and tau.

    Args:
        ksi (float): The probability level for which the scale factor is calculated.
        tau (float, optional): The tau value used in scale factor calculation. 
                               Defaults to 0.75.
        N (int, optional): An unused parameter, included for potential future use.
                            Defaults to 1000.

    Returns:
        float: The calculated scale factor.
    """
    return norm.ppf(tau) / norm.ppf(ksi)

def tail_cor(X1, X2, ksi, tau=0.75):
    """
    Calculate the "TailCor" between two data sets.

    Args:
        X1 (array-like): The first data set.
        X2 (array-like): The second data set.
        ksi (float): The probability level used in the tail correlation calculation.
        tau (float, optional): The percentile to use for IQR calculation in tail 
                               correlation. Defaults to 0.75.

    Returns:
        float: The calculated tail correlation between X1 and X2.
    """
    Y1 = (X1 - np.percentile(X1, 50)) / IQR(X1)
    Y2 = (X2 - np.percentile(X2, 50)) / IQR(X2)
    
    cov_matrix = np.cov(Y1, Y2)
    sign_of_corr = np.sign(cov_matrix[0][1])
    Z = 1/np.sqrt(2) * (Y1 + sign_of_corr * Y2)
    return sg(ksi, 0.75) * IQR(Z, ksi)
