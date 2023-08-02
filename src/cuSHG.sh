#!/bin/bash

# This file contains a set of instructions to run the main file cuOPO.cu.
# Use this file to perform simulations in bulk. This means, for example, 
# systematically varying the input power, the cavity reflectivity, etc.

TIMEFORMAT='It took %0R seconds.' 
time { 

clear                                # Clear screen
rm *.dat                             # This removes all .dat files in the current folder. Comment this line for safe.
rm *.txt                             # This removes all .txt files in the current folder. Comment this line for safe. 
rm cuSHG                           # This removes a previuos executable file (if it exist)


########################################################################################################################################
# -----Compilation-----
# Notice there are 2 preprocessor variables in this compilation that are useful to set
# the regime and the number of equations used in the simulations:
# a) For set the crystal use: -DSPPLT (or PPLN). You must to define it!!. 
# b) For set temperature profile use: -DTHERMAL.
 
nvcc cuSHG.cu --gpu-architecture=sm_60 -lcufftw -lcufft -o cuSHG 
# FOLDERSIM="Simulations_Waist_variable"
# FOLDERSIM="Simulations_prueba"
FOLDERSIM="Simulations_P_variable"

# nvcc cuSHG.cu -DTHERMAL --gpu-architecture=sm_60 -lcufftw -lcufft -o cuSHG
# FOLDERSIM="Simulations_cuSHG"


# There are three flags specific for CUDA compiler:
# --gpu-architecture=sm_75: please check your GPU card architecture (Ampere, Fermi, Tesla, Pascal, Turing, Kepler, etc) 
#                           to set the correct number sm_XX. This code was tested using a Nvidia GeForce GTX 1650 card (Turing
#                           architecture). 
# 				    Please visit https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
#                           to set the proper flag according to your GPU card.
# -lcufftw -lcufft        : flags needed to perform cuFFT (the CUDA Fourier transform)
########################################################################################################################################
# The variables defined below (ARGX) will be passed as arguments to the main file 
# cuOPO.cu on each execution via the argv[X] instruction.

ARG1=(20)     # Pump power                         (ARG1)
ARG2=(31)     # Beam waist (um)                   (ARG2)
ARG3=(57.18 ) # 1st peltier temperature     (ARG3)
ARG4=(57.18 ) # 2nd peltier temperature   (ARG4)

SOLS=1       # Save Only Last Slice (FALSE=0,TRUE=1)


# x=$(awk 'BEGIN{for(i=20;i<=30;i=i+0.1)print i}')
n=$(awk 'BEGIN{for(i=1;i<=20;i=i+1)print i}')
# t1=$(awk 'BEGIN{for(i=49.9;i<=54.02;i=i+0.1)print i}')
# Each for-loop span over one or more values defined in the previous arguments. 
# 		for N in $x
# 		do

# for (( p=0; p<${#ARG1[@]}; p++ ))
# do
for N in $n
do
	for (( x=0; x<${#ARG2[@]}; x++ ))
	do
# 	for WA in $x
# 	do
		for (( t1=0; t1<${#ARG3[@]}; t1++ ))
		do
# 		for TOVEN1 in $t1
# 		do

# 			N=${ARG1[$p]}
			printf "\nPower			= ${N} W\n" 
			
			WA=${ARG2[$x]}
			printf "\nBeam waist	= ${WA} \n" 
			
			TOVEN1=${ARG3[$t1]}
			printf "\nT peltier 1		= ${TOVEN1} ºC\n"
			
# 			TOVEN2=${ARG4[$t1]}
			TOVEN2=${TOVEN1}
			printf "\nT peltier 2	= ${TOVEN2} ºC\n"
			
			printf "\nMaking directory...\n"
			FOLDER="sPPLT_P_${N}_W_T1_${TOVEN1}_WAIST_${WA}"
			FILE="sPPLT_P_${N}_W_T1_${TOVEN1}_WAIST_${WA}.txt"
			
			printf "Bash execution and writing output file...\n\n"
			./cuSHG $N $WA $TOVEN1 $TOVEN2 $SOLS | tee -a $FILE
			
			printf "Bash finished!!\n\n" 
			mkdir $FOLDER
			mv *.dat $FOLDER"/"
			mv *.txt $FOLDER"/"

		done
	done
done


if [ -d "$FOLDERSIM" ]; then
	echo "Moving simulations in ${FOLDERSIM}..."
	mv sPPLT* $FOLDERSIM"/" 
else

	mkdir $FOLDERSIM
	echo "Creating and moving simulations in ${FOLDERSIM}..."
	mv sPPLT* $FOLDERSIM"/" 
fi

# mv -v $FOLDERSIM"/" ..

}
