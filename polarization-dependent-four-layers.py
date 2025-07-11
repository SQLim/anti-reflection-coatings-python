# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 11:08:39 2025
Calculate reflection of s- and p-polarizations at various angles of incidences 
(incidence angles chosen must be less than angle for total internal reflection) 
@author: SQLim

# References:
# [1] https://iopscience.iop.org/article/10.1088/0034-4885/23/1/301/pdf
# [2] Macleod 2018 Thin film optical filters 4th edition
"""

import numpy as np
from matplotlib import pyplot as plt

###############
# user inputs #
###############

# design wavelength of AR coating
wavelen = 1064.0 # nm [float]

# refractive index of air and substrate substrate
n_0 = 1.0  # air [float]
n_s = 2.4  # substrate [float or np.array]

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
t_3 = t_1
t_4 = t_2

# choose incidence angles to plot
thetas_deg = np.arange(start=0., stop=20., step=5.) 


###############
# Definitions #
###############

# plot set up
fig0, ax0 = plt.subplots() # for R vs incidence angle plot (no AR coat)
fig, ax = plt.subplots()   # for R vs wavelength plot (4-layer AR coat)
wavelens = np.arange(start=250.0, stop=1460.0, step=10.0)

# refractive index array setup
if type(n_s) == np.ndarray:
	n_0 = np.array( [ n_0 ] * len(wavelens) )

def r_s(n_i, n_f, theta_i):
	""" 
	Fresnel's  reflection coefficient s-polarization.
	"""
	num = n_i * np.cos(theta_i) - n_f * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 )
	den = n_i * np.cos(theta_i) + n_f * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 )
	return num/den

def r_p(n_i, n_f, theta_i):
	""" 
	Fresnel's reflection coefficient p-polarization.
	"""
	num = n_i * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 ) - n_f * np.cos(theta_i)
	den = n_i * np.sqrt( 1 - ( n_i / n_f * np.sin(theta_i) )**2 ) + n_f * np.cos(theta_i)
	return num/den


################
# Calculations #
################

### angular dependence of R 
# assumes equal p- and s-polarized light
thetas0_deg = np.arange(start=0., stop=90., step=0.1)
thetas0 = thetas0_deg / 180 * np.pi
R_substrate = ( r_s(n_0, n_s, thetas0)**2 + r_p(n_0, n_s, thetas0)**2 ) / 2

ax0.plot(thetas0_deg, R_substrate)
ax0.set(xlabel='theta_i (deg)', ylabel='(R_s+R_p)/2', title='uncoated substrate')
fig0.show()


### R with 4-layer AR coating (SiOx/SiNx)
for j, theta_deg in enumerate(thetas_deg):	
	# convert to radians
	theta = theta_deg / 180 * np.pi

	# calculate Fresnel's reflection coefficients at each interface
	# s-polarization
	r01_s = r_s(n_0, n_1, theta) # real array, air-SiOx interface
	r12_s = r_s(n_1, n_2, theta) # real array, SiOx-SiNx interface
	r23_s = r_s(n_2, n_1, theta) # real array, SiNx-SiOx interface
	r34_s = r12_s                # real array, SiOx-SiNx interface
	r45_s = r_s(n_2, n_s, theta) # real array, SiNx-substrate interface
	
	# p-polarization
	r01_p = r_p(n_0, n_1, theta) # real array, air-SiOx interface
	r12_p = r_p(n_1, n_2, theta) # real array, SiOx-SiNx interface
	r23_p = r_p(n_2, n_1, theta) # real array, SiNx-SiOx interface
	r34_p = r12_p                # real array, SiOx-SiNx interface
	r45_p = r_p(n_2, n_s, theta) # real array, SiNx-substrate interface
	
	# initialize empty list for reflectance calculations
	R_multi_s = []   
	R_multi_p = []
	
	# transfer matrix method calculation
	for i, wave in enumerate(wavelens): 
		delta1 = 2*np.pi*t_1*n_1[i]/wave
		delta2 = 2*np.pi*t_2*n_2[i]/wave
		delta3 = 2*np.pi*t_3*n_1[i]/wave
		delta4 = 2*np.pi*t_4*n_2[i]/wave
		
		exp1 = np.exp(1j*delta1)
		exp2 = np.exp(1j*delta2)
		exp3 = np.exp(1j*delta3)
		exp4 = np.exp(1j*delta4)
		
		M1_s = np.array([[exp1, r12_s[i] / exp1], [r12_s[i]*exp1, 1 / exp1]])
		M2_s = np.array([[exp2, r23_s[i] / exp2], [r23_s[i]*exp2, 1 / exp2]])
		M3_s = np.array([[exp3, r34_s[i] / exp3], [r34_s[i]*exp3, 1 / exp3]])
		M4_s = np.array([[exp4, r45_s[i] / exp4], [r45_s[i]*exp4, 1 / exp4]])
		
		M1_p = np.array([[exp1, r12_p[i] / exp1], [r12_p[i]*exp1, 1 / exp1]])
		M2_p = np.array([[exp2, r23_p[i] / exp2], [r23_p[i]*exp2, 1 / exp2]])
		M3_p = np.array([[exp3, r34_p[i] / exp3], [r34_p[i]*exp3, 1 / exp3]])
		M4_p = np.array([[exp4, r45_p[i] / exp4], [r45_p[i]*exp4, 1 / exp4]])
		
		# For reflection off air-SiO2 interface without multilayer interference:
		# amplitude of incoming E field = 1, amplitude of reflected = r(n_0, n_1)
		E1_s = np.array([[1], [r01_s[i]]])
		E1_p = np.array([[1], [r01_p[i]]])
		
		E5_s = M4_s@M3_s@M2_s@M1_s@E1_s
		E5_p = M4_p@M3_p@M2_p@M1_p@E1_p
			
		num_s = E5_s[1][0]*E5_s[1][0].conj()
		den_s = E5_s[0][0]*E5_s[0][0].conj()
		num_p = E5_p[1][0]*E5_p[1][0].conj()
		den_p = E5_p[0][0]*E5_p[0][0].conj()
		
		if np.isreal(num_s) and np.isreal(den_s):
			R_multi_s += [num_s/den_s]
		else: 
			print('Error: complex square not real at {wave:.2f} nm!')
			break
		
		if np.isreal(num_p) and np.isreal(den_p):
			R_multi_p += [num_p/den_p]
		else: 
			print('Error: complex square not real at {wave:.2f} nm!')
			break
	
	R_multi_s = np.array(R_multi_s)
	R_multi_p = np.array(R_multi_p)
	
	ax.plot(wavelens, R_multi_s, label=f's-pol {theta_deg:.0f}', c=f'C{j}')
	ax.plot(wavelens, R_multi_p, label=f'p-pol {theta_deg:.0f}', c=f'C{j}', ls='dashed')
	
ax.set(xlabel='wavelength (nm)', ylabel='R', title=f'SiO/SiN stack: d1={t_1:.2f}, d2={t_2:.2f}, d3={t_3:.2f}, d4={t_4:.2f} nm')
ax.legend()

fig.tight_layout()
fig.show()
