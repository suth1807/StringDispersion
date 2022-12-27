# Author : suthambhara@gmail.com

import math
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import scipy.signal as signal


def compute_dispersion_filter_LS(f0: float, B: float, df: float, ord: int,
                                 fs: float = 44100.0) -> np.array:
    """
    Compute an all pass filter satisfying group delays of a stiff string
    . This uses a non linear least squares algorithm for optimization

    Arguments:

    f0:  Fundamental frequency of the string
    B:   Dispersion coefficient
    df:  Design frequency. Max frequency until which fidelity is enforced. 
    ord: Order of the desired filter 
    fs:  Sampling rate

    Returns:
    SOS of dispersion filter with order ord
    """
    N = math.floor(df / f0)
    f = np.arange(0, N+1, 1, dtype='float')
    f *= f0
    tau0 = fs/f0
    pd0 = tau0/math.sqrt(1 + B*(df/f0)*(df/f0))
    mu0 = pd0/(1 + B*(df/f0)*(df/f0))

    grp_delays = tau0 / np.sqrt(1.0 + B * (f/f0) * f / f0)
    grp_delays *= 1.0 / (1.0 + B * (f/f0) * f / f0)
    grp_delays -= mu0
    grp_delays = np.concatenate([np.flip(grp_delays), grp_delays[1:]])

    freqs = np.arange(-N, N+1, 1)
    discrete_freqs = freqs * 2.0 * math.pi * f0 / fs
    # order of the filter is ord. This means we have one magnitude and one phase per filter in the cascade.
    # Additionally, we allow for a constant group delay overall
    num_var = 2 * ord + 1
    # Initial values to the solver
    initial_theta = np.linspace(0, discrete_freqs[-1], ord)
    initial_rhos = np.zeros(ord)
    initial_rhos[:] = 0.5
    initial_vals = np.zeros(num_var)
    initial_vals[0:ord] = initial_theta
    initial_vals[ord:-1] = initial_rhos
    initial_vals[-1] = 0.0

    lower_bounds = np.zeros_like(initial_vals)
    upper_bounds = np.zeros_like(initial_vals)
    lower_bounds[:] = -np.inf
    upper_bounds[:] = np.inf
    # Poles have to be within the unit circle for stability. Add as condition to the solver
    lower_bounds[ord:-1] = 0.0
    upper_bounds[ord:-1] = 0.9999

    res = least_squares(__compute_residual, initial_vals, bounds=(lower_bounds, upper_bounds), ftol=0.001, xtol=0.001, gtol=0.001,
                        args=(ord, N, discrete_freqs, grp_delays))

    # Compute section of Bi-Quads from poles
    thetas = res.x[0:ord]
    rhos = res.x[ord:-1]
    A = np.ones(ord)
    B = -2.0 * rhos * np.cos(thetas)
    C = rhos * rhos

    temp = np.stack([C, B, A, A, B, C])
    sos = np.transpose(temp)
    return sos


def compute_dispersion_filter(f0: float, B: float,
                              df: float, beta: float, fs: float = 44100.0) -> np.array:
    """
    This is a port from MATLAB and implements the algorithm in the paper
    "Robust, Efficient Design of Allpass Filters for
    Dispersive String Sound Synthesis" by Abel, Valimaki and Smith

    Arguments:

    f0:  Fundamental frequency of the string
    B:   Dispersion coefficient
    df:  Design frequency. Max frequency until which fidelity is enforced. 
    beta: Smoothing value of the fit (see paper)
    fs:  Sampling rate

    Returns:
    SOS of dispersion filter 

    """
    tau0 = fs/f0
    pd0 = tau0/math.sqrt(1 + B*(df/f0)*(df/f0))
    mu0 = pd0/(1 + B*(df/f0)*(df/f0))
    phi0 = 2*math.pi*(df/fs)*pd0 - mu0*2*math.pi*df/fs

    nap = math.floor(phi0/(2*math.pi))
    phik = math.pi*np.arange(1, 2*nap, 2)

    etan = np.arange(0, nap)/(1.2*nap) * df

    pdn = tau0/np.sqrt(1.0 + B*(etan/f0)*(etan/f0))
    taun = pdn/(1 + B*(etan/f0)*(etan/f0))
    phin = 2*math.pi*(etan/fs)*pdn
    theta = fs/(2*math.pi) * (phik - phin +
                              (2*math.pi/fs)*etan*taun) / (taun - mu0)

    delta = np.diff(np.concatenate([[-theta[0]], theta])) * 0.5

    cc = np.cos(theta * 2*math.pi/fs)
    beta_arr = np.zeros_like(cc)
    # We may not need to do this and could use just one value of beta everywhere. 
    # allows different beta per segment for manual tuning
    beta_arr[:] = beta
    eta = (1 - beta_arr*np.cos(delta * 2*math.pi/fs))/(1 - beta_arr)
    alpha = np.sqrt(eta*eta - 1) - eta
    A = np.ones(nap)
    B = 2 * alpha * cc
    C = alpha * alpha
    temp = np.stack([C, B, A, A, B, C])
    sos = np.transpose(temp)
    return sos


def compute_dispersion(L: float, d: float, Y: float, f_0: float, rho) -> float:
    """
    Compute stiff string dispersion coefficient given physical parameters

    Arguments:
    L : Length of the string in meters (e.g. 613e-3)
    d : Diameter in meters (e.g. 0.381e-3)
    Y : Young's modulus in Pascals (e.f. 90e6 for Bronze)
    f_0: Fundamental frequency
    rho: Density of the material( not density per meter ) in Kg/M^3 (e.g. 8.77e3 for Bronze)

    Returns:
    Dispersion coefficient B
    """
    # L = 613e-3  # Length of string in m
    # d = 0.381e-3  # diameter of string in m
    # Y = 90e6  # Young's modulus in Pascals
    # f_0 = 138.0  # fundamental frequency
    # rho = 8.77e3  # density in kg/m3

    T = L*L * f_0*f_0 * rho * math.pi * d * d

    B = (math.pi * math.pi * math.pi * Y * d * d * d * d) / (64 * L * L * T)
    return B

def __compute_residual(x, ord, N, freqs, grp_delays):
    """
    INTERNAL ROUTINE TO COMPUTE RESIDUAL. DO NOT USE
    """
    theta = x[0:ord]
    rho = x[ord:-1]
    C = x[2*ord]
    residuals = np.zeros_like(grp_delays)
    for i, omega in enumerate(freqs):
        # Compute for both theta and -theta
        c1 = np.cos(omega - theta)
        c2 = np.cos(omega + theta)
        a1 = (1.0 - rho * rho) / (1.0 + rho * rho - 2.0 * rho * c1)
        a2 = (1.0 - rho * rho) / (1.0 + rho * rho - 2.0 * rho * c2)
        residuals[i] = np.sum(a1) + np.sum(a2) - C - grp_delays[i]
    return residuals

if __name__ == '__main__':
    # Examples
    example1 = compute_dispersion_filter(138.0, 0.00004, 7500, 0.85)
    example2 = compute_dispersion_filter_LS(
        277.0, 0.0000145, 15000, 6, 44100.0)
