# Python port of Matlab code given at https://ccrma.stanford.edu/~jos/pasp/Matlab_Passive_Reflectance_Synthesis.html
# suthambhara@gmail.com
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from typing import Tuple
import math

import numpy as np

from scipy.signal import lfilter
from scipy.signal import tf2sos

def find_bridge_admittance_filter(fs: float, resonances : np.array, bandwidths: np.array, uniform_loss_factor=0.99, use_jos_method_2: bool = False):
    """
        Port of Matlab code available at https://ccrma.stanford.edu/~jos/pasp/Matlab_Passive_Reflectance_Synthesis.html
    """
    nsec = resonances.shape[0]

    R = np.exp(-math.pi*bandwidths/fs);     # Pole radii
    theta = 2*math.pi*resonances/fs;     # Pole angles
    poles = R * np.exp(1j*theta)
    A1 = -2*R*np.cos(theta);  # 2nd-order section coeff
    A2 = R * R;              # 2nd-order section coeff
    denoms = np.stack([np.ones(nsec), A1, A2], 1)
    A = np.zeros(2*nsec + 1)
    A[0] = 1.0
    
    for i in range(nsec):
        # polynomial multiplication = FIR filtering:
        A = lfilter(denoms[i,:],1,A)
    
    g = uniform_loss_factor

    B = g*np.flip(A) # Flip(A)/A = desired allpass

    Badm = A - B
    Aadm = A + B
    Badm = Badm/Aadm[0]
    Aadm = Aadm/Aadm[0]

    if use_jos_method_2:
        B = np.zeros(2*nsec+1)
        impulse = np.zeros(2*nsec+1) 
        impulse[0] = 1
        for i in range(nsec):
            # filter() does polynomial division here:
            B = B + lfilter(A,denoms[i,:],impulse)
        
        # add a near-zero at dc 
        B = lfilter(np.array([1, -0.995]),1,B)

        Badm = B
        Aadm = A

    sos = tf2sos(Badm,Aadm)

    return sos
   

if __name__ == '__main__':

    #Bridge resonances of a Sarod
    resonances = np.array([220, 350])
    bandwidth  = np.array([30,  30])
    sos = find_bridge_admittance_filter(44100.0, resonances, bandwidth, uniform_loss_factor=0.99, use_jos_method_2=True)

    w, h = signal.sosfreqz(sos, 2048, fs=44100.0)
    s = np.s_[0:100]
    fig, ax1 = plt.subplots(2)
    h_log = 20 * np.log10(np.maximum(abs(h), 1e-5))
    ax1[0].plot(w[s], np.abs(h_log[s]), 'b')
    ax1[1].plot(w[s], np.angle(h[s]) * 180 / math.pi, 'b')
    plt.show()

    pass