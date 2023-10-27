/*---------------------------------------------------------------------------*/
// * This file contains two functions that save files in .dat extension
// * 1 - saveFileVectorReal()    : save real vectors
// * 2 - saveFileVectorComplex() : save complex vectors

// Inputs:
// - Vector   : vector to save
// - N        : vector size
// - Filename : name of the saved file

// * Functions ending in GPU take as an argument device vectors
/*---------------------------------------------------------------------------*/


#ifndef _FILESCUH
#define _FILESCUH

#pragma once


void saveVectorReal (real_t *Vector, const int N, std::string Filename)
{
	std::ofstream myfile;
	myfile.open(Filename);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy] << "\n";
	myfile.close();
	
	return;
	
}


void saveVectorComplex (complex_t *Vector, const int N, std::string Filename)
{
	std::ofstream myfile;
	std::string extension_r = "_r.dat", extension_i = "_i.dat";
	myfile.open(Filename+extension_r);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy].x << "\n";
	myfile.close();
	myfile.open(Filename+extension_i);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy].y << "\n";
	myfile.close();
	
	return;
	
}


void saveMatrixReal (real_t *Vector, std::string Filename)
{
	std::ofstream myfile;
	std::string extension = ".dat";
	myfile.open(Filename+extension);
	for (int iy = 0; iy < NY; iy++){
		for (int ix = 0; ix < NX; ix++)
			myfile << std::setprecision(20) << Vector[iy*NX+ix] << "\t";
		myfile << "\n"; 
	}
	myfile.close();
	
	return;
	
}


void saveMatrixComplex (complex_t *Vector, std::string Filename)
{
	std::ofstream myfile;
	std::string filenamer = "_r.dat", filenamei = "_i.dat";
	myfile.open(Filename+filenamer);
	for (int iy = 0; iy < NY; iy++){
		for (int ix = 0; ix < NX; ix++)
			myfile << std::setprecision(20) << Vector[iy*NX+ix].x << "\t";
		myfile << "\n"; 
	}
	myfile.close();
	myfile.open(Filename+filenamei);
	for (int iy = 0; iy < NY; iy++){
		for (int ix = 0; ix < NX; ix++)
			myfile << std::setprecision(20) << Vector[iy*NX+ix].y << "\t";
		myfile << "\n"; 
	}
	myfile.close();
	
	return;
	
}


void saveVectorComplexGPU (complex_t *Vector_gpu, const int N, std::string Filename)
{
	uint nBytes = N*sizeof(complex_t);
	complex_t *Vector = (complex_t*)malloc(nBytes);
	CHECK(cudaMemcpy(Vector, Vector_gpu, nBytes, cudaMemcpyDeviceToHost));
	saveVectorComplex ( Vector, N, Filename );
	free(Vector);
	
	return;
	
}


void saveMatrixRealGPU (real_t *Vector_gpu, std::string Filename)
{
	real_t *Vector = (real_t*)malloc(nBytes2Dr);
	CHECK(cudaMemcpy(Vector, Vector_gpu, nBytes2Dr, cudaMemcpyDeviceToHost));
	saveMatrixReal ( Vector, Filename );
	free(Vector);
	
	return;
	
}


void saveMatrixComplexGPU (complex_t *Matrix_gpu, std::string Filename)
{
	// 	uint nBytes2D = NX * NY * sizeof(complex_t);
	complex_t *Matrix = (complex_t*)malloc(nBytes2Dc);
	CHECK(cudaMemcpy(Matrix, Matrix_gpu, nBytes2Dc, cudaMemcpyDeviceToHost));
	saveMatrixComplex ( Matrix, Filename );
	free(Matrix);

	return;
	
}


void readMatrixReal (real_t *Vector, std::string Filename)
{
	std::ifstream myfile;
	myfile.open(Filename); 	
	std::cout.precision(20);
	for (int iy = 0; iy < NY; iy++){
		for (int ix = 0; ix < NX; ix++)
			myfile >> Vector[iy*NX+ix];
	}
	myfile.close();
	
	return;
	
}


void readFileMatrixRealGPU (real_t *Vector_gpu, std::string Filename)
{
	
	unsigned long int nBytes2Dr = NX*NY*sizeof(real_t);
	real_t *Vector_cpu = (real_t*)malloc(nBytes2Dr);
	
	std::ifstream myfile;
	myfile.open(Filename); 	
	std::cout.precision(20);
	for (int iy = 0; iy < NY; iy++){
		for (int ix = 0; ix < NX; ix++)
			myfile >> Vector_cpu[iy*NX+ix];
	}
	myfile.close();
	CHECK(cudaMemcpy(Vector_gpu, Vector_cpu, nBytes2Dr, cudaMemcpyHostToDevice));	
	free(Vector_cpu);
	
	return;
	
}


void saveTensorReal ( real_t *Tensor, std::string Filename )
{
	
	std::ofstream myfile;
	std::string extension = ".dat";
	
	for (uint iz = 0; iz < NZ; iz++){
		myfile.open(Filename+"_"+std::to_string(iz)+extension);
		for (uint iy = 0; iy < NY; iy++){
			for (uint ix = 0; ix < NX; ix++){
				myfile << Tensor[IDX(ix,iy,iz)] << "\t";
			}
			myfile << "\n"; 
		}
		myfile.close();
	}
	
	return ;
	
}

void saveTensorRealGPU ( real_t *Tensor_gpu, std::string Filename )
{
	
	unsigned long int nBytes3D = TSIZE * sizeof(real_t);
	real_t *Tensor = (real_t*)malloc(nBytes3D);
	CHECK(cudaMemcpy(Tensor, Tensor_gpu, nBytes3D, cudaMemcpyDeviceToHost));
	saveTensorReal ( Tensor, Filename );
	free(Tensor);
	
	return ;

	
}

#endif // -> #ifdef _FILESCUH
