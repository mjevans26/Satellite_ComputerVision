import numpy as np
from scipy.special import gamma

def gamma_pdf(x, a, b):
    """calculate the pdf of a gamma distribution defined by shape a and scale b"""
    denom = gamma(a)*(b**a)
    num = (x**(a-1))*(np.exp(-1*x/b))
    pd = num/denom
    return pd

def lognormal_pdf(x, u, v):
    """calculate the pdf of a lognormal distribution defined by mean u and variance v"""
    sd = np.sqrt(v)
    const = (pi*2)**0.5
    first = 1/(sd*const)
    edenom = v*2
    enum = ((np.log(x) - u)**2)*-1
    second = np.exp(enum/edenom)/x
    pd = first*second
    return pd

