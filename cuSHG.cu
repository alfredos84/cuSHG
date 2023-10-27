// * Author name: Alfredo Daniel Sanchez
// * email:               alfredo.daniel.sanchez@gmail.com

#include "headers/Libraries.h"						// Required libraries

// Datatypes for real and complex numbers
using real_t = float;
using complex_t = cufftComplex;

// Spatial grid number of points
__constant__ const uint NX	= 128;				// Number of pints in X
__constant__ const uint NY		= 128;				// Number of pints in Y
__constant__ const uint NZ	= 100;				// Number of points in Z
__constant__ const uint NXY	= NX * NY;		// Number of points in Z
__constant__ const uint TSIZE	= NX*NY*NZ;	// Number of points in full 3D-grid

// Memory size for vectors and matrices
const size_t nBytes2Dr = sizeof(real_t) * NXY;			// real 2D 
const size_t nBytes2Dc = sizeof(complex_t) * NXY;		// complex 2D
const size_t nBytes3Dr = sizeof(real_t) * TSIZE;			// real 3D 

const uint BLKX      = 16;	// Block dimensions for kernel functions
const uint BLKY      = 16;

#include "headers/PackageLibraries.h"			// Required package libraries

int main(int argc, char *argv[]){
	std::cout << "#######---SHG efficiency calculator---#######\n\n" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////
	// 1. Set GPU and timing	
	
	int dev = cudaGetDevice(&dev);	// Set up device (GPU)
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cout << "\n\nUsing Device " << dev << ": GPU " << deviceProp.name << std::endl;
	cudaSetDevice(dev);
	
	double iStart = seconds();	// Timing code
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// 2. Set input parameters
	
	// Pump
	real_t Power		= atof(argv[1]);	// Pump power [W]
	real_t waist		= atof(argv[2]);	// Initial beam waist radius [μm]
	real_t focalpoint	= 0.5*LZ;			// Focal point at z = Lcr/2[μm]
	
	// Thermal properties and oven temperatures
	real_t Temp			= atof(argv[3]);	// Initial temperature [ºC]
	real_t T_inf		= 25;						// Temp environment
	real_t Tpeltier	= atof(argv[3]);		// Peltier 1
	
	// Set 'save_only_last = true' for users interested only in calculating the output efficiency. 
	// Otherwise, 'solver->run(/*args*/)' save the fields in the NZ slices, increasing the number of output files.
	bool save_only_last = atoi(argv[4]);
	bool save_temperature = atoi(argv[5]); // save output temperature tensor
	
	// Print parameters values
	bool printvalues = true;
	if(printvalues){
		std::cout << "\n\nPump power =  " << Power << " W" << std::endl;
		std::cout << "Beam waist = " << waist << " μm" << std::endl;
		std::cout << "Focal distance = " << focalpoint << " μm" << std::endl;
		std::cout << "Phase-matching temperature =  " << Temp << " ºC" << std::endl;
		std::cout << "Environment temperature =  " << T_inf << " ºC" << std::endl;
		std::cout << "Peltier 1 temperature =  " << Tpeltier << " ºC\n" << std::endl;
		// Display crystal properties
		GetCrystalProp();
	}
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	// 3. Model execution
	
	Solver *solver = new Solver;	
	solver->run( Power, waist, focalpoint, Temp, T_inf, Tpeltier, save_only_last, save_temperature );
	delete solver;
	///////////////////////////////////////////////////////////////////////////////////
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// 4. Reset the GPU and finish simulation timing
	
	cudaDeviceReset();		std::cout << "\n\nDevice reset" << std::endl;
	
	double iElaps = seconds() - iStart;	// finish timing
	TimingCode( iElaps); // print time
	
	// Open a file in append mode
	std::ofstream outputFile("GPUTime_NZ_"+std::to_string(NZ)+"_N_"+std::to_string(NX)+"_T.txt", std::ios::app);
	
	if (outputFile.is_open()) {
		// Write data to the end of the file
		outputFile << iElaps << std::endl;
		
		// Close the file
		outputFile.close();
		
		std::cout << "Data successfully written to the end of the file." << std::endl;
	} else {
		std::cerr << "Unable to open the file for writing." << std::endl;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	return 0;
	
}
