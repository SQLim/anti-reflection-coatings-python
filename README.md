# anti-reflection-coatings-python
## Abstract
Python scripts that calculate the reflectance spectrum of substrates coated with thin film(s) using the transfer matrix technique.  

These scripts were written to design basic anti-reflection coatings for use in physics experiments.

## Intro
### average-sp-polarization-one-or-two-layers.py
The script calculates the average R for single- and two-layer coatings at normal incidence only. It can be easily generalized to calculate R for more than two layers. 

### polarization-dependent-four-layers.py
The script calculates the R for s- and p-polarized light for a four-layer thin film stack for various incidence angles (only tested for angles less than the total internal reflection angle). 


## Usage
Required user definitions:
- design wavelength
- substrate and thin film refractive indices (a few examples are provided within the scipt)
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


## Notes
A Python script that allows the use of complex refractive indices also exists but in a less well-commented form and may be made available upon reasonable request. 


## References:
1. https://iopscience.iop.org/article/10.1088/0034-4885/23/1/301/pdf
2. Macleod 2018 Thin film optical filters 4th edition
