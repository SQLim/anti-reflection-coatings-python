# anti-reflection-coatings-python
## Abstract
Python scripts that calculate the reflectance spectrum of substrates coated with thin film(s) using the transfer matrix method.  

These scripts were written to help design basic anti-reflection coatings for use in physics experiments.

## Intro
### average-sp-polarization-one-or-two-layers.py
The script calculates the average reflectance (R) for single- and two-layer coatings at normal incidence only. It can be easily generalized to calculate R for more than two layers. 

### reflectance_simulator.py
The script calculates the reflectance (R) for s- and p-polarized light for multilayer (N-layers) thin film stack of the form substrate/H/L/H/L/.../H/L/air over a range of incidence angles (only tested for angles less than the total internal reflection angle). H and L here refers to a high and low index material (e.g., TiO$_2$ and SiO$_2$).

## Usage
### average-sp-polarization-one-or-two-layers.py
Required user definitions:
- design wavelength
- substrate and thin film refractive indices as a list/array (a few examples are provided within the scipt)
- thin film thickness

Please define substrate and thin film material by a refractive index array (1-D) with the following properties:
- length: 121
- start wavelength: 250 nm
- stop wavelength: 1450 nm
- wavelength increment: 10 nm
- assume extinction coefficient is zero (i.e., refrative index, n, is purely real)

Default settings in script:
- Substrate: diamond (n_s = 2.4)
- Thin film 1: SiOx (silicon oxide)
- Thin film 2: SiNx (silicon nitride)

To change substrate type or thin film material, simply replace the refractive indeces, n_s, n_1, n_2, with the desired values.

### reflectance_simulator.py
Required user definitions:
- substrate and thin film refractive indices. These should be external data files (e.g., *.txt).
- thicknesses for each layer in the N-layer stack
- angle of incidence range and resolution
- wavelength range and resolution

Requirements for substrate and thin film refractive indices data:
- Three columns in this order: Wavelength (nm), n, k
- Wavelength in units of nm
- If k data is not available, set this column to zeros


## Notes
Work in progress: a Python script that searches for the optimal layer thicknesses that results in an anti-reflective coating in a specified wavelength range. 

## References:
1. https://iopscience.iop.org/article/10.1088/0034-4885/23/1/301/pdf
2. Macleod 2018 Thin film optical filters 4th edition
