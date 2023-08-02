/*---------------------------------------------------------------------------*/
// * This file contains the class Solver. 
// * The class contains functions to solve the Split-Step Fourier method (SSMF)
// * needed to calculate the electric fields evolution through the nonlinear crystal.
// * 
// * In particular, this file should be used when only two equation describes the 
// * problem, e.g., parametric down-convertion or second-harmonic generation.
// * Only two frequencies are involved in theses problems.
/*---------------------------------------------------------------------------*/


#ifndef _SOLVERCUHCUH
#define _SOLVERCUHCUH

#pragma once

__global__ void KernelComplexProduct ( complex_t *C, complex_t *A, complex_t *B )
{
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( (idx < NX) and (idy < NY) ){
		C[IDX(idx,idy,idz)] = A[IDX(idx,idy,idz)] * B[IDX(idx,idy,idz)] ;
	}
	
	return ;
	
}

/** Computes the nonlinear part: dA/dz=i.κ.Ax.Ay.exp(i.Δk.L) and saves the result in dAx (x,y are different fields) */

__global__ void dAdz( complex_t *dPump, complex_t *dSignal, complex_t *Pump, complex_t *Signal, real_t *Tfinal, real_t *DK, real_t *temperature, real_t *zz, real_t increment, uint *slice )
{
		
	real_t PI	= 3.14159265358979323846;     // pi
	complex_t Im   = make_cuComplex(0.0f, 1.0f); // imaginary number
	real_t Tpm = *temperature;
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	#ifdef THERMAL
	uint s = *slice;
	if( idx < NX and idy < NY){
		dPump[IDX(idx,idy,idz)] = (Im*4*PI*deff/(n(lp, Tfinal[IDX(idx,idy,s)])*lp) * Signal[IDX(idx,idy,idz)] * CpxConj(Pump[IDX(idx,idy,idz)]) * CpxExp(-(DK[IDX(idx,idy,idz)]))) ;
// 		dPump[IDX(idx,idy,idz)] = make_cuComplex(0.0,0.0);
		dSignal[IDX(idx,idy,idz)] = (Im*2*PI*deff/(n(ls, Tfinal[IDX(idx,idy,s)])*ls) * Pump[IDX(idx,idy,idz)] * Pump[IDX(idx,idy,idz)] * CpxExp(DK[IDX(idx,idy,idz)])) - 0.5*beta_crs*0.5*C*EPS0*n(ls, Tfinal[IDX(idx,idy,s)]) * CpxAbs2(Signal[IDX(idx,idy,idz)]) * Signal[IDX(idx,idy,idz)] ;
	}
	#else	
	real_t z = *zz + dz*increment;
	if( idx < NX and idy < NY){
// 		dPump[IDX(idx,idy,idz)] = (Im*4*PI*deff/(n(lp, Tpm)*lp) * Signal[IDX(idx,idy,idz)] * CpxConj(Pump[IDX(idx,idy,idz)]) * CpxExp(-z*DK[IDX(idx,idy,idz)])) ;
		dPump[IDX(idx,idy,idz)] =make_cuComplex(0.0,0.0);
		dSignal[IDX(idx,idy,idz)] = (Im*2*PI*deff/(n(ls, Tpm)*ls) * Pump[IDX(idx,idy,idz)] * Pump[IDX(idx,idy,idz)] * CpxExp(+z*DK[IDX(idx,idy,idz)])) - 0.5*beta_crs*0.5*C*EPS0*n(ls,Tpm)*CpxAbs2(Signal[IDX(idx,idy,idz)]) * Signal[IDX(idx,idy,idz)] ;
	}
	#endif
	
	
	return ;
	
}

/** This kernel computes the final sum after appling the Rounge-Kutta algorithm */
__global__ void rk4( complex_t *Pump, complex_t *Signal, complex_t *k1p, complex_t *k1s, complex_t *k2p, complex_t *k2s,complex_t *k3p, complex_t *k3s,complex_t *k4p, complex_t *k4s )
{
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		Pump[IDX(idx,idy,idz)] = Pump[IDX(idx,idy,idz)] + (k1p[IDX(idx,idy,idz)] + 2*k2p[IDX(idx,idy,idz)] + 2*k3p[IDX(idx,idy,idz)] + k4p[IDX(idx,idy,idz)]) * dz / 6;
		Signal[IDX(idx,idy,idz)] = Signal[IDX(idx,idy,idz)] + (k1s[IDX(idx,idy,idz)] + 2*k2s[IDX(idx,idy,idz)] + 2*k3s[IDX(idx,idy,idz)] + k4s[IDX(idx,idy,idz)]) * dz / 6;
	}
	
	return ;
	
}

/** Computes a linear combination Ax + s.kx and saves the result in aux_x */
__global__ void LinealCombination( complex_t *auxp, complex_t *auxs, complex_t *Pump, complex_t *Signal, complex_t *kp, complex_t *ks, double s )
{
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;	
	
	if( idx < NX and idy < NY){
		auxp[IDX(idx,idy,idz)] = Pump[IDX(idx,idy,idz)] + kp[IDX(idx,idy,idz)] * s;
		auxs[IDX(idx,idy,idz)] = Signal[IDX(idx,idy,idz)] + ks[IDX(idx,idy,idz)] * s;
	}
	
	return ;
	
}


class Solver
{	
public:	
	complex_t *k1p, *k2p, *k3p, *k4p, *k1s, *k2s, *k3s, *k4s;
	complex_t *auxp, *auxs;	
	
	// Constructor
	__host__ __device__ Solver( ){
		//* RK4 (kx) and auxiliary (aux) GPU vectors 
		cudaMalloc((void **)&k1p, nBytes2Dc );
		cudaMalloc((void **)&k2p, nBytes2Dc );
		cudaMalloc((void **)&k3p, nBytes2Dc );
		cudaMalloc((void **)&k4p, nBytes2Dc );
		cudaMalloc((void **)&k1s, nBytes2Dc );
		cudaMalloc((void **)&k2s, nBytes2Dc );
		cudaMalloc((void **)&k3s, nBytes2Dc );
		cudaMalloc((void **)&k4s, nBytes2Dc );
		cudaMalloc((void **)&auxp, nBytes2Dc );
		cudaMalloc((void **)&auxs, nBytes2Dc );
				
		complex_t *k1p_cpu = new complex_t[NX*NY];
		complex_t *k2p_cpu = new complex_t[NX*NY];
		complex_t *k3p_cpu = new complex_t[NX*NY];
		complex_t *k4p_cpu = new complex_t[NX*NY];
		complex_t *k1s_cpu = new complex_t[NX*NY];
		complex_t *k2s_cpu = new complex_t[NX*NY];
		complex_t *k3s_cpu = new complex_t[NX*NY];
		complex_t *k4s_cpu = new complex_t[NX*NY];
		complex_t *auxp_cpu = new complex_t[NX*NY];
		complex_t *auxs_cpu = new complex_t[NX*NY];

		cudaMemcpy(k1p, k1p_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k2p, k2p_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k3p, k3p_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k4p, k4p_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k1s, k1s_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k2s, k2s_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k3s, k3s_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(k4s, k4s_cpu, nBytes2Dc, cudaMemcpyHostToDevice );		
		
		delete[] k1p_cpu;	delete[] k1s_cpu;
		delete[] k2p_cpu;	delete[] k2s_cpu;
		delete[] k3p_cpu;	delete[] k3s_cpu;
		delete[] k4p_cpu;	delete[] k4s_cpu;
		delete[] auxp_cpu;	delete[] auxs_cpu;		
	}
	
	// Destructor
	__host__ __device__ ~Solver() {
		cudaFree(k1p);	cudaFree(k2p);	cudaFree(k3p);		cudaFree(k4p);
		cudaFree(k1s);	cudaFree(k2s);	cudaFree(k3s);		cudaFree(k4s);	
		cudaFree(auxs);		cudaFree(auxp);		
	}
	
	void Diffraction( Efields *E );
	void SSFM( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, real_t z, uint slice );
	void CWES( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, bool save_only_last );
	void RK4Solver( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, real_t z, uint slice );
// 	void RK4Solver( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, real_t rk4step, real_t z, uint slice );
	
	
};


void Solver::Diffraction ( Efields *E )
{
	// Parameters for kernels 2D
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
	
	// Set plan for cuFFT 2D//
	cufftHandle plan2D;	cufftPlan2d(&plan2D, NX, NY, CUFFT_C2C );
	
	cufftExecC2C(plan2D, (complex_t *)E->Pump, (complex_t *)E->PumpQ, CUFFT_FORWARD);
	CHECK(cudaDeviceSynchronize());
	cufftExecC2C(plan2D, (complex_t *)E->Signal, (complex_t *)E->SignalQ, CUFFT_FORWARD);
	CHECK(cudaDeviceSynchronize());	
	
	KernelComplexProduct<<<grid2D, block2D>>> (E->AuxQ, E->PropPump, E->PumpQ);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(E->PumpQ, E->AuxQ, nBytes2Dc, cudaMemcpyDeviceToDevice));
	
	KernelComplexProduct<<<grid2D, block2D>>> (E->AuxQ, E->PropSignal, E->SignalQ);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(E->SignalQ, E->AuxQ, nBytes2Dc, cudaMemcpyDeviceToDevice));
	
	cufftExecC2C(plan2D, (complex_t *)E->PumpQ, (complex_t *)E->Pump, CUFFT_INVERSE);
	CHECK(cudaDeviceSynchronize());
	CUFFTscale<<<grid2D, block2D>>>(E->Pump);
	CHECK(cudaDeviceSynchronize());		
	cufftExecC2C(plan2D, (complex_t *)E->SignalQ, (complex_t *)E->Signal, CUFFT_INVERSE);
	CHECK(cudaDeviceSynchronize());
	CUFFTscale<<<grid2D, block2D>>>(E->Signal);
	CHECK(cudaDeviceSynchronize());
	
	cufftDestroy(plan2D);
	
	return ;
}


void Solver::RK4Solver( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, real_t z, uint slice )
{
	//* This function apply the fourth-order Runge-Kutta method
	
	uint *slice_device;	cudaMalloc((void**)&slice_device, sizeof(uint));
	cudaMemcpy(slice_device, &slice, sizeof(uint), cudaMemcpyHostToDevice);
	real_t *z_device;	cudaMalloc((void**)&z_device, sizeof(real_t));
	cudaMemcpy(z_device, &z, sizeof(real_t), cudaMemcpyHostToDevice);
	real_t *Tpm_device;	cudaMalloc((void**)&Tpm_device, sizeof(real_t));
	cudaMemcpy(Tpm_device, &Tpm, sizeof(real_t), cudaMemcpyHostToDevice);
// 	real_t *rk4step_device;	cudaMalloc((void**)&rk4step_device, sizeof(real_t));
// 	cudaMemcpy(rk4step_device, &rk4step, sizeof(real_t), cudaMemcpyHostToDevice);	
	
// 	real_t factor = 0.5; //*rk4step_device;
	
	// Parameters for kernels 2D
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);

	dAdz<<<grid2D,block2D>>>( this->k1p, this->k1s, E->Pump, E->Signal, T->Tfinal,  DK->DK, Tpm_device, z_device, 0.0, slice_device );
	CHECK(cudaDeviceSynchronize()); 
	
	LinealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, E->Pump, E->Signal, this->k1p, this->k1s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	dAdz<<<grid2D,block2D>>>( this->k2p, this->k2s, this->auxp, this->auxs, T->Tfinal, DK->DK, Tpm_device, z_device, 0.5*0.5, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	LinealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, E->Pump, E->Signal, this->k2p, this->k2s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	dAdz<<<grid2D,block2D>>>( this->k3p, this->k3s, this->auxp, this->auxs, T->Tfinal, DK->DK, Tpm_device, z_device, 0.5*0.5, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	LinealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, E->Pump, E->Signal, this->k3p, this->k3s, 1.0 );
	CHECK(cudaDeviceSynchronize());   
	dAdz<<<grid2D,block2D>>>( this->k4p, this->k4s, this->auxp, this->auxs, T->Tfinal, DK->DK, Tpm_device, z_device, 1.0*0.5, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	rk4<<<grid2D,block2D>>>( E->Pump, E->Signal, this->k1p, this->k1s, this->k2p, this->k2s, this->k3p, this->k3s,this->k4p, this->k4s );
	CHECK(cudaDeviceSynchronize());
	
	cudaFree((void *) z_device); cudaFree((void *) slice_device);  cudaFree((void *) Tpm_device);
	
	return ;
	
}

void Solver::SSFM( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, real_t z, uint slice )
{
	
	// Split-Step Fourier Method (SSFM) in the scheme (NL/2 - L - NL/2)
	RK4Solver ( E, T,  DK, Tpm, z, slice ); // Runge-Kutta 4 for dz/2
	Diffraction ( E ) ; // Diffraction for dz
	RK4Solver ( E, T,  DK, Tpm, z, slice ); // Runge-Kutta 4 for dz/2
	return ;
}

void Solver::CWES( Efields *E, Tfield *T, PhaseMatching *DK, real_t Tpm, bool save_only_last )
{	
	real_t z = 0.0;
	uint s = 0;
	while( s < NZ) {
		if(save_only_last){// Save fields in the last slice (save_only_last = true)
			if( s == NZ-1 ){
				std::cout << "Saving only last slice" << std::endl;
				SaveMatrixComplexGPU ( E->Pump, "Pump_out" );
				SaveMatrixComplexGPU ( E->Signal, "Signal_out" );
			}
		}
		else{
			if( s <= NZ-1 ){// Save fields in every slice (save_only_last = false)
				std::cout << "Saving slice #" << s << std::endl;
				SaveMatrixComplexGPU ( E->Pump, "Pump_"+std::to_string(s) );
				SaveMatrixComplexGPU ( E->Signal, "Signal_"+std::to_string(s) );
			}				
		}
			// Split-Step Fourier Method (SSFM) for the z-th step
		SSFM( E, T,  DK, Tpm, z, s );
		
		#ifdef THERMAL
		if( s > 0 ){
			T->UpDateQ( E,  s );
			DK->SetAccumulatedDK( T, s );
		}
		#endif			
		z+=dz; s++;// next slice
	}

	return ;		
}


#endif // -> #ifdef _SOLVERCUH
