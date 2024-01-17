# cuSHG package

is a CUDA-based toolkit for simulating focused Gaussian beams second-harmonic generation (SHG) effciency using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media. 
CUDA programming allows us to implement parallel computing in order to speed up calculations that typically require a considerable computational demand.

The provided software implements a solver for the CWEs including diffraction terms, linear absorption and nonlinear absorption and thermal evolution. The package is restricted to use it in the continuos-wave (CW) regime.
This code implements a scheme based on Split-Step Fourier Method (SSFM).

For running this package is necessary to have a GPU in your computer and installed the CUDA drivers and the CUDA-TOOLKIT as well. 
To install the CUDA driver on a Linux system please visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html


## Setup and execution

To run simulations using the package clone this project typing in a terminal
```
git clone https://github.com/alfredos84/cuSHG.git
```
Once the project was cloned, the user will find a parent folder `cuSHG` containing two other
- `src`: contains the main file `cuSHG.cu`, the headers folder, and the bash file `cuSHG.sh` used to compile and execute the package by passing several simulations parameters.
- `src/headers`: this folder contains the headers files needed to execute the package.

### Bash file `src/cuSHG.sh`

The bash file is mainly used to massively perform simulations by passing the main file different parameters such as pump power, beam waist, oven temparture, etc.
Before starting the user has to allow the system execute the bash file. To do that type in the terminal
```
chmod 777 cuSHG.sh # enable permissions for execution
```

Finally, to execute the file execute the following command line
```
./cuSHG.sh         # execute the files
```

When finished a new folder named in the bash variable `FOLDERSIM` will have been created containing the output files.

In the `cuSHG.sh` file you will find two different the command lines for the compilation:
```
nvcc cuSHG.cu -DTHERMAL --gpu-architecture=sm_75 -lcufftw -lcufft -o cuSHG
```
that includes the thermal calculations, and
```
nvcc cuSHG.cu -DTHERMAL --gpu-architecture=sm_75 -lcufftw -lcufft -o cuSHG
```
that does not include thermal calculations (this mode is faster than the first one).
Currently, there are available two crystals, namely, MgO:PPLN and MgO:sPPLT. 
The type of crystal is set in the header file `Crystal.h`, where the MgO:sPPLT nonlinear crystal is set by default. This file should be modified according to users' needs.


The compilation flag `--gpu-architecture=sm_75` is related to the GPU architecture (see below more information about this point). 
The flags `-lcufftw` and `-lcufft` tell the compiler to use the `CUFFT library` that performs the Fourier transform on GPU .

Finally, the execution is done using the command line in the `cuSHG.sh` file is
```
./cuSHG <ARGUMENTS_TO_PASS>
```
where `$ARGx` and others are variables externaly passed to the main file `cuSHG.cu`.
It was written in this way to make easier performing simulations massively.

### Outputs

This package returns a set of `.dat` files with the fundamental (pump) and SH (signal) electric fields, separated into real and imaginary parts.

### GPU architecture
Make sure you know your GPU architecture before compiling and running simulations. For example, pay special attention to the sm_75 flag defined in the provided `cuSHG.sh` file. 
That flag might not be the same for your GPU since it corresponds to a specific architecture. For instance, I tested this package using two different GPUs:
1. Nvidia Geforce MX250: architecture -> Pascal -> flag: sm_60
2. Nvidia Geforce GTX1650: architecture -> Turing -> flag: sm_75

Please check the NVIDIA documentation in https://docs.nvidia.com/cuda/


### Contact me
For any questions or queries, do not hesitate to contact the developer by writing to alfredo.daniel.sanchez@gmail.com or alfredo.sanchez@icfo.eu.
