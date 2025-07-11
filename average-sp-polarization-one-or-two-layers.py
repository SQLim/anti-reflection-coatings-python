# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 11:08:39 2025
Calculate reflection (average s- and p-polarizations) at normal incidence 
@author: SQLim
"""

# References:
# [1] https://iopscience.iop.org/article/10.1088/0034-4885/23/1/301/pdf
# [2] Macleod 2018 Thin film optical filters 4th edition

import numpy as np
from matplotlib import pyplot as plt

###############
# user inputs #
###############

# design wavelength of AR coating
wavelen = 532 # nm

# refractive index of air and substrate
n_0 = 1.0  # air
n_s = 2.42 # substrate

# refractive index of thin films (assume purely real between 250-1450 nm)
# low index coating (n_1) should be closer to air (n_0)
# high index coating (n_2) should be closer to substrate (n_s)
n_1 = np.array([1.5074, 1.5024, 1.498, 1.4942, 1.4908, 1.4878, 1.4851, 1.4827, 1.4806, 1.4787, 1.4769, 1.4753, 1.4738, 1.4725, 1.4713, 1.4701, 1.4691, 1.4681, 1.4672, 1.4663, 1.4656, 1.4648, 1.4641, 1.4635, 1.4629, 1.4623, 1.4618, 1.4613, 1.4608, 1.4603, 1.4599, 1.4595, 1.4591, 1.4587, 1.4584, 1.458, 1.4577, 1.4574, 1.4571, 1.4568, 1.4565, 1.4563, 1.456, 1.4558, 1.4555, 1.4553, 1.4551, 1.4549, 1.4546, 1.4544, 1.4542, 1.454, 1.4539, 1.4537, 1.4535, 1.4533, 1.4531, 1.453, 1.4528, 1.4527, 1.4525, 1.4523, 1.4522, 1.452, 1.4519, 1.4518, 1.4516, 1.4515, 1.4513, 1.4512, 1.4511, 1.4509, 1.4508, 1.4507, 1.4505, 1.4504, 1.4503, 1.4502, 1.45, 1.4499, 1.4498, 1.4497, 1.4496, 1.4494, 1.4493, 1.4492, 1.4491, 1.449, 1.4489, 1.4487, 1.4486, 1.4485, 1.4484, 1.4483, 1.4482, 1.4481, 1.4479, 1.4478, 1.4477, 1.4476, 1.4475, 1.4474, 1.4473, 1.4471, 1.447, 1.4469, 1.4468, 1.4467, 1.4466, 1.4465, 1.4464, 1.4462, 1.4461, 1.446, 1.4459, 1.4458, 1.4457, 1.4455, 1.4454, 1.4453, 1.4452]) # SiOx PECVD
n_2 = np.array([2.2714, 2.238, 2.2112, 2.1893, 2.171, 2.1555, 2.1422, 2.1306, 2.1204, 2.1114, 2.1034, 2.0963, 2.0898, 2.084, 2.0787, 2.0739, 2.0695, 2.0654, 2.0617, 2.0583, 2.0551, 2.0522, 2.0494, 2.0469, 2.0445, 2.0423, 2.0402, 2.0383, 2.0364, 2.0348, 2.0331, 2.0316, 2.0301, 2.0287, 2.0275, 2.0262, 2.025, 2.0239, 2.0228, 2.0218, 2.0209, 2.02, 2.0191, 2.0182, 2.0174, 2.0166, 2.0158, 2.0152, 2.0145, 2.0138, 2.0132, 2.0125, 2.0119, 2.0114, 2.0108, 2.0102, 2.0097, 2.0092, 2.0087, 2.0083, 2.0078, 2.0074, 2.0069, 2.0065, 2.0061, 2.0056, 2.0053, 2.0049, 2.0045, 2.0042, 2.0038, 2.0034, 2.0031, 2.0028, 2.0024, 2.0021, 2.0018, 2.0015, 2.0012, 2.0009, 2.0006, 2.0003, 2.0001, 1.9997, 1.9995, 1.9992, 1.9989, 1.9987, 1.9985, 1.9981, 1.9979, 1.9977, 1.9974, 1.9972, 1.997, 1.9967, 1.9965, 1.9963, 1.996, 1.9958, 1.9956, 1.9953, 1.9951, 1.9949, 1.9947, 1.9945, 1.9943, 1.994, 1.9938, 1.9936, 1.9934, 1.9932, 1.993, 1.9927, 1.9926, 1.9924, 1.9922, 1.992, 1.9918, 1.9916, 1.9914])

# thickness of thin films
# generally, we want to satisfy the quater-wave condition for destructive interference:
# t = wavelen / 4 / n_1@wavelen
t_1 = 100 # nm
t_2 = 100 # nm 

###############
# Definitions #
###############

# plot set up
fig, ax = plt.subplots()
wavelens = np.arange(start=250.0, stop=1460.0, step=10.0)

def r(ni, nf):
	""" 
	Fresnel's coefficient at normal incidence (phi = 0).
	(reflection is this coefficient squared, R=r^2, r is real).
	(Ref. [1] equations 8 and 10)
	"""
	return (ni-nf)/(ni+nf)

idx = 0
for i, wave in enumerate(wavelens): 
    if round(wavelen/10) == wave/10:
        idx = i

wavelen_index_rounded = idx
n_1_i = n_1[wavelen_index_rounded]
n_2_i = n_2[wavelen_index_rounded]

################
# Calculations #
################

### R without thin film coating (air-substrate interface)
# R = Fresnel's coefficient squared
# refractive index is real so it is simply r-squared
R_substrate = np.array( [r(n_0, n_s)**2] * len(wavelens) ) 


### R with thin film coating
# calculated using transfer matrix method
# from Ref. [1] equation 16 (or 38) and 15 
# assumes normal incidence, phi = 0, cos(phi)=1 in equation 15

##### single SiOx coating
r1 = r(n_0, n_1)                            # real array, Fresnel's coefficient at air-coating1 interface
r2 = r(n_1, n_s)                            # real array, Fresnel's coefficient at coating1-substrate interface

R_single = []
for i, wave in enumerate(wavelens): 
	delta1 = 2*np.pi*t_1*n_1_i/wave
	
	exp_pos1 = np.exp(1j*delta1)
	exp_neg1 = np.exp(-1j*delta1)

	M1 = np.array([[exp_pos1, r2[i]*exp_neg1], [r2[i]*exp_pos1, exp_neg1]])
	
	# For reflection off air-SiO2 interface:
	# amplitude of incoming E field = 1, amplitude of reflected = r(n_0, n_1)
	E1_vector = np.array([[1], [r1[i]]]) 
	E2_vector = M1@E1_vector
	
	num = E2_vector[1][0]*E2_vector[1][0].conj()
	den = E2_vector[0][0]*E2_vector[0][0].conj()
	
	if np.isreal(num) and np.isreal(den):
		R_single += [num/den] 
	
	else: 
		print('Error: complex square not real at {wave:.2f} nm!')
		break

R_single = np.array(R_single)


##### R with SiOx/SiNx coating 
r01 = r(n_0, n_1)    # real array, Fresnel's coefficient at air-coating1 interface
r12 = r(n_1, n_2)    # real array, Fresnel's coefficient at coating1-2 interface
r23 = r(n_2, n_s)    # real array, Fresnel's coefficient at coating2-substrate interface

R_double = []
for i, wave in enumerate(wavelens): 
	delta1 = 2*np.pi*t_1*n_1_i/wave
	delta2 = 2*np.pi*t_2*n_2_i/wave
	
	exp_pos1 = np.exp(1j*delta1)
	exp_neg1 = np.exp(-1j*delta1)
	exp_pos2 = np.exp(1j*delta2)
	exp_neg2 = np.exp(-1j*delta2)
	
	M1 = np.array([[exp_pos1, r12[i]*exp_neg1], [r12[i]*exp_pos1, exp_neg1]])
	M2 = np.array([[exp_pos2, r23[i]*exp_neg2], [r23[i]*exp_pos2, exp_neg2]])
	
	# For reflection off air-SiO2 interface:
	# amplitude of incoming E field = 1, amplitude of reflected = r(n_0, n_1)
	E1_vector = np.array([[1], [r01[i]]])
	E3_vector = M2@M1@E1_vector
	
	num = E3_vector[1][0]*E3_vector[1][0].conj()
	den = E3_vector[0][0]*E3_vector[0][0].conj()
	
	if np.isreal(num) and np.isreal(den):
		R_double += [num/den] 

	else: 
		print('Error: complex square not real at {wave:.2f} nm!')
		break

R_double = np.array(R_double)


########
# plot #
########

ax.plot(wavelens, R_substrate)
ax.plot(wavelens, R_single)
ax.plot(wavelens, R_double)

ax.set(xlabel='wavelength (nm)', ylabel='R')
ax.legend(['uncoated substrate', f'SiOx {t_1:.2f} nm', f'SiOx {t_1:.2f} nm,\nSiNx {t_2:.2f} nm'])

fig.tight_layout()
fig.show()



