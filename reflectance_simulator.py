# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:06:03 2025

Calculates reflectance spectrum given a multi-layer stack alternating between
a high index and low index material using the transfer matrix method.

Users need to specify the thickness of each layer in the stack and provide 
refractive index data for the low index, high index and substrate materials.
This data file should contain these three columns in this order:
    Wavelength (nm), n, k
The first row is assumed to be the header row and will be ommited when 
the data is loaded.

Refractive index of air is assumed to be constant and real: n = 1.0. 

@author: SQLim
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# --- User definitions ---

# data file locations
root_path = "C:\\Users\\UserName\\Documents\\refractive index data\\"
substrate_file = "n_Si.txt"
nL_file = "n_SiO2.txt"
nH_file = "n_TiO2.txt"
seperator = '\t'

# Plot settings
SUBSTRATE = "Si"
HIGH_INDEX = "TiO₂"
LOW_INDEX = "SiO₂"
WAVELENGTH_MIN = 350.0  # wavelength-axis starting point (nm)
WAVELENGTH_MAX = 1200.0
WAVELENGTH_RES = 1.0    # wavelength plotting resolution (nm)

# choose incidence angles to plot
ANGLES_START = 0.0
ANGLES_STOP = 20.0
ANGLES_STEP = 10.0

# layer thicknesses, starts with the layer exposed to air,
# ends with layer in contact with substrate
d_array = np.array([120.0, 78.0])



# --- Fetching and Parsing Material Data ---

# Function to convert refractive index data to an cubic interpolation function
def parse_data(file, delimiter):
    """
    Reads a data file with columns for Wavelength(nm), n, and k,
    and returns cubic interpolation functions for n and k.
    """
    try:
        data = np.loadtxt(root_path+file, skiprows=1, delimiter=delimiter)
        # Create interpolation functions for the refractive index (n) and extinction coefficient (k)
        n_func = interp1d(data[:, 0], data[:, 1], kind='cubic', bounds_error=False, fill_value="extrapolate")
        k_func = interp1d(data[:, 0], data[:, 2], kind='cubic', bounds_error=False, fill_value="extrapolate")
        return n_func, k_func
    except Exception as e:
        print(f"Error parsing file content: {e}")
        return None, None

# Load user-provided refractive index data 
n_LOW_func, k_LOW_func = parse_data(nL_file, '\t')
n_HIGH_func, k_HIGH_func = parse_data(nH_file, '\t')
n_substrate_func, k_substrate_func = parse_data(substrate_file, '\t')

# Check if data was loaded correctly
if not all([n_LOW_func, n_HIGH_func, n_substrate_func]):
    raise ValueError("Failed to load one or more material data files.")



# --- Internal definitions ---

# total number of layers
N = len(d_array)

# wavelength array
lambdas = np.arange(start=WAVELENGTH_MIN, stop=WAVELENGTH_MAX, step=WAVELENGTH_RES)

# set up array of incidence angles
thetas_deg = np.arange(start=ANGLES_START, stop=ANGLES_STOP+ANGLES_STEP, step=ANGLES_STEP) 

# refractive indices
n_0 = 1.0                                                          # air
n_1 = n_LOW_func(lambdas) + 1j * k_LOW_func(lambdas)               # low index material
n_2 = n_HIGH_func(lambdas) + 1j * k_HIGH_func(lambdas)             # high index material
n_sub = n_substrate_func(lambdas) + 1j * k_substrate_func(lambdas) # substrate

# Fresnel's reflection coefficients
# reflectance_s = r_s ** 2
def r_s(n_i, n_f, theta_i):
	""" 
	Fresnel's  reflection coefficient s-polarized light.
	"""
	num = n_i * np.cos(theta_i) - n_f * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 )
	den = n_i * np.cos(theta_i) + n_f * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 )
	return num/den

def r_p(n_i, n_f, theta_i):
	""" 
	Fresnel's reflection coefficient p-polarized light.
	"""
	num = n_i * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 ) - n_f * np.cos(theta_i)
	den = n_i * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 ) + n_f * np.cos(theta_i)
	return num/den



# --- Optical calculations ---

# initialize plot
fig, ax = plt.subplots()

# calculate reflectance spectra over full range of angle of incidence
for j, theta_deg in enumerate(thetas_deg):	
	# convert to radians
    theta = theta_deg / 180 * np.pi
    
    # calculate angle in each layer
    theta_1 = np.arcsin( (n_0 / n_1) * np.sin(theta) )
    theta_2 = np.arcsin( (n_1 / n_2) * np.sin(theta_1) )
    theta_sub = np.arcsin( (n_2 / n_sub) * np.sin(theta_2) )
	
	# calculate Fresnel's reflection coefficients at each interface
	# s-polarization
    r0L_s = r_s(n_0, n_1, theta)     # real array, air-L interface
    rLH_s = r_s(n_1, n_2, theta_1)   # real array, L-H interface
    rHL_s = r_s(n_2, n_1, theta_2)   # real array, H-L interface
    rHd_s = r_s(n_2, n_sub, theta_2) # real array, H-diamond interface
	
	# p-polarization
    r0L_p = r_p(n_0, n_1, theta)     # real array, air-L interface
    rLH_p = r_p(n_1, n_2, theta_1)   # real array, L-H interface
    rHL_p = r_p(n_2, n_1, theta_2)   # real array, H-L interface
    rHd_p = r_p(n_2, n_sub, theta_2) # real array, H-diamond interface
	
	# initialize empty lists for reflectance calculations
    R_multi_s = []
    R_multi_p = []
	
	# Construct transfer matrix through each layer
    for i, wave in enumerate(lambdas): 
        # initialize empty lists for transfer matrices
        M_s_array = []
        M_p_array = []
        
        for k in range(N):
            # low index material exponent
            if (k%2 == 0):
                delta = 2*np.pi*d_array[k]*n_1[i]*np.cos(theta_1[i])/wave
                rs = rLH_s
                rp = rLH_p
            # last layer contacting diamond
            elif (k%2 == 1) and (k == N-1):
                delta = 2*np.pi*d_array[k]*n_2[i]*np.cos(theta_2[i])/wave
                rs = rHd_s
                rp = rHd_p
            # high index material exponent
            else:
                delta = 2*np.pi*d_array[k]*n_2[i]*np.cos(theta_2[i])/wave 
                rs = rHL_s
                rp = rHL_p
            
            # exponential term
            exp = np.exp(1j*delta)
            
            # build array of transfer matrices
            # there should be a total of N transfer matrices at the end
            M_s_array += [ np.array([[exp, rs[i] / exp], [rs[i]*exp, 1 / exp]]) ]
            M_p_array += [ np.array([[exp, rp[i] / exp], [rp[i]*exp, 1 / exp]]) ]
		
		# Set up initial electric field vector 
        # For reflection off the first air-L interface, the amplitude of 
        # the incoming E field = 1, so the amplitude of the reflected = r(n_0, n_1, theta)
        E1_s = np.array([[1], [r0L_s[i]]])
        E1_p = np.array([[1], [r0L_p[i]]])
        
        # Now apply the transfer matrices sequentially to the E-field vector
        E_s = E1_s
        E_p = E1_p
        # iterate from air to substrate
        for m in range(N):
            E_s = M_s_array[m] @ E_s
            E_p = M_p_array[m] @ E_p

        # calculate reflectance
        num_s = E_s[1][0]*E_s[1][0].conj()
        den_s = E_s[0][0]*E_s[0][0].conj()
        num_p = E_p[1][0]*E_p[1][0].conj()
        den_p = E_p[0][0]*E_p[0][0].conj()
        R_multi_s += [num_s/den_s * 100] 
        R_multi_p += [num_p/den_p * 100] 
    
    # cast reflectance list into array type
    R_multi_s = np.array(R_multi_s)
    R_multi_p = np.array(R_multi_p)
	
    # plot
    ax.plot(lambdas, R_multi_s, label=f'S-pol {theta_deg:.0f}$\degree$', c=f'C{j}')
    ax.plot(lambdas, R_multi_p, label=f'P-pol {theta_deg:.0f}$\degree$', c=f'C{j}', ls='dashed')
    ax.plot(lambdas, (R_multi_p + R_multi_s) / 2, label=f'R_avg {theta_deg:.0f}$\degree$', c='black', ls='dashed')

# plot aesthetics 
ax.set(xlabel='Wavelength (nm)', ylabel='Reflectance (%)', title=f'{HIGH_INDEX}/{LOW_INDEX} on {SUBSTRATE}')
ax.legend()
fig.tight_layout()
fig.show()
