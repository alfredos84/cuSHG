// * Author name: Alfredo Daniel Sanchez
// * email:               alfredo.daniel.sanchez@gmail.com


#include "headers/Libraries.h"						// Required libraries

// Datatypes for real and complex numbers
using real_t = float;
using complex_t = cuFloatComplex;

// Spatial grid number of points
__constant__ const uint ARXY	= 1;					// Crystal aspect ratio (X/Y)
__constant__ const uint N		= 256;				// Number of pints in X-Y       
__constant__ const uint NX		= N*ARXY;		// Number of pints in X
__constant__ const uint NY		= N;				// Number of pints in Y
__constant__ const uint NZ		= 100;			// Number of points in Z
__constant__ const uint NXY	=  NX * NY;		// Number of points in Z
__constant__ const uint TSIZE	= NX*NY*NZ;	// Number of points in full 3D-grid

// Memory size for vectors and matrices
const size_t nBytes2Dr =  sizeof(real_t) * NXY;			// real 2D 
const size_t nBytes2Dc =  sizeof(complex_t) * NXY;	// complex 2D
const size_t nBytes3Dr =  sizeof(real_t) * TSIZE;			// real 3D 

const uint BLKX      = 16;	// Block dimensions for kernel functions
const uint BLKY      = 16;

#include "headers/PackageLibraries.h"			// Required package libraries

int main(int argc, char *argv[]){
	std::cout << "#######---SHG efficiency calculator---#######\n\n" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////
	double iStart = seconds();				// Timing code
	
	int dev = cudaGetDevice(&dev); 		// Set up device (GPU)
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	std::cout << "\n\nUsing Device " << dev << ": GPU " << deviceProp.name << std::endl;
	std::cout << "Compute capability: " << deviceProp.major << "-" << deviceProp.minor << "\n\n\n" << std::endl;
	CHECK(cudaSetDevice(dev));
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	//* Input parameters
	
	//* Pump
	real_t Power			= atof(argv[1]);	// Pump power [W]
	real_t waist			= atof(argv[2]);	// inicial beam waist radius [μm]
	real_t focalpoint	= 0.5*LZ;			// focal point [μm]
	
	//* Thermal properties and oven temperatures
	real_t Tpm		= atof(argv[3]);		// Phase matching temperature [ºC]
	real_t T_inf		= 25;					// T environment
	real_t Tpeltier1	= atof(argv[3]);		// Peltier 1
	real_t Tpeltier2	= atof(argv[4]);		// Peltier 2
	
	//* Print parameters values
	std::cout << "\n\nPump power =  " << Power << " W" << std::endl;
	std::cout << "Beam waist = " << waist << " μm" << std::endl;
	std::cout << "Focal distance = " << focalpoint << " μm" << std::endl;
	std::cout << "Phase-matching temperature =  " << Tpm << " ºC" << std::endl;
	std::cout << "Environment temperature =  " << T_inf << " ºC" << std::endl;
	std::cout << "Peltier 1 temperature =  " << Tpeltier1 << " ºC" << std::endl;
	std::cout << "Peltier 2 temperature =  " << Tpeltier2 << " ºC\n" << std::endl;
	
	// Display crystal properties
	GetCrystalProp();
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	// Set 'save_only_last = true' for users interested only in calculating the output efficiency. 
	// Otherwise, 'EvolutionInCrystal()' save the fields in the NZ slices, increasing the number of output files.
	bool save_only_last = atoi(argv[5]);
	bool save_temperature = false; // save output temperature tensor
	
	std::cout << "\nModel execution...\n" << std::endl;	
	EvolutionInCrystal( Power, waist, focalpoint, 
						Tpm, T_inf, Tpeltier1, Tpeltier2, 
						save_only_last, save_temperature );
	
	///////////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////////////////
	CHECK(cudaDeviceReset());	//* Reset the GPU
	std::cout << "\n\nDevice reset" << std::endl;
	
	//* Finish simulation timing	
	double iElaps = seconds() - iStart;	
	TimingCode( iElaps);
	////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}
