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

__global__ void kernelComplexProduct ( complex_t *C, complex_t *A, complex_t *B )
{	// Product of complex numbers in GPU
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( (idx < NX) and (idy < NY) ){
		C[IDX(idx,idy,idz)] = A[IDX(idx,idy,idz)] * B[IDX(idx,idy,idz)] ;
	}
	
	return ;
	
}

#ifdef THERMAL
__global__ void dAdz( complex_t *dPump, complex_t *dSignal, complex_t *Pump, complex_t *Signal, real_t *Tfinal, real_t *DKint, real_t increment, uint *slice )
{	// Nonlinear term: dA/dz=i.κ.Ax.Ay.exp(i.Δk.L) and saves the result in dPump/dSignal (x,y are different fields)
	
	real_t PI	= 3.14159265358979323846;     // pi
	complex_t Im   = make_cuComplex(0.0f, 1.0f); // negative imaginary number -i
	
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	real_t s = (real_t)*slice;
	
	if( idx < NX and idy < NY)
	{
		dPump[IDX(idx,idy,idz)] = Im*1.0*PI*dQ/(n(lp, Tfinal[IDX(idx,idy,s)])*lp) * Signal[IDX(idx,idy,idz)] * CpxConj(Pump[IDX(idx,idy,idz)]) * CpxExp(-(DKint[IDX(idx,idy,idz)]*(dz*increment+s*dz)/(s+1.0))) - 0.5*alpha_crp*Pump[IDX(idx,idy,idz)] ;
		// 		dPump[IDX(idx,idy,idz)] = make_cuComplex(0.0f, 0.0f);
		dSignal[IDX(idx,idy,idz)] = Im*0.5*PI*dQ/(n(ls,Tfinal[IDX(idx,idy,s)])*ls)*Pump[IDX(idx,idy,idz)]*Pump[IDX(idx,idy,idz)]*CpxExp(+(DKint[IDX(idx,idy,idz)]*(dz*increment+s*dz)/(s+1.0))) - 0.5*(alpha_crs + beta_crs*0.5*EPS0*C*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)])) * Signal[IDX(idx,idy,idz)] ;
	}
	
	return ;
	
}
#else
__global__ void dAdz( complex_t *dPump, complex_t *dSignal, complex_t *Pump, complex_t *Signal, real_t *DK, real_t *temperature, real_t *zz, real_t increment )
{	// Nonlinear term: dA/dz=i.κ.Ax.Ay.exp(i.Δk.L) and saves the result in dPump/dSignal (x,y are different fields)
	
	real_t PI	= 3.14159265358979323846;     // pi
	complex_t Im   = make_cuComplex(0.0f, 1.0f); // negative imaginary number -i
	
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	real_t z = *zz + dz*increment;
	real_t Temp = *temperature;
	
	if( idx < NX and idy < NY){
		dPump[IDX(idx,idy,idz)] = Im*1*PI*dQ/(n(lp, Temp)*lp) * Signal[IDX(idx,idy,idz)] * CpxConj(Pump[IDX(idx,idy,idz)]) * CpxExp(-z*DK[IDX(idx,idy,idz)]) - 0.5*alpha_crp*Pump[IDX(idx,idy,idz)] ;
		// 		dPump[IDX(idx,idy,idz)] = make_cuComplex(0.0f, 0.0f);
		dSignal[IDX(idx,idy,idz)] = Im*0.5*PI*dQ/(n(ls, Temp)*ls) * Pump[IDX(idx,idy,idz)] * Pump[IDX(idx,idy,idz)] * CpxExp(+z*DK[IDX(idx,idy,idz)]) - 0.5*(alpha_crs + beta_crs*0.5*C*EPS0*n(ls, Temp)*CpxAbs2(Signal[IDX(idx,idy,idz)])) * Signal[IDX(idx,idy,idz)] ; 
	}
	
	
	return ;
	
}
#endif

__global__ void kernelRK4( complex_t *Pump, complex_t *Signal, complex_t *k1p, complex_t *k1s, complex_t *k2p, complex_t *k2s,complex_t *k3p, complex_t *k3s,complex_t *k4p, complex_t *k4s )
{	// Final sum after appling the Rounge-Kutta algorithm
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		Pump[IDX(idx,idy,idz)] = Pump[IDX(idx,idy,idz)] + (k1p[IDX(idx,idy,idz)] + 2*k2p[IDX(idx,idy,idz)] + 2*k3p[IDX(idx,idy,idz)] + k4p[IDX(idx,idy,idz)]) * dz / 6;
		Signal[IDX(idx,idy,idz)] = Signal[IDX(idx,idy,idz)] + (k1s[IDX(idx,idy,idz)] + 2*k2s[IDX(idx,idy,idz)] + 2*k3s[IDX(idx,idy,idz)] + k4s[IDX(idx,idy,idz)]) * dz / 6;
	}
	
	return ;
	
}


__global__ void linealCombination( complex_t *auxp, complex_t *auxs, complex_t *Pump, complex_t *Signal, complex_t *kp, complex_t *ks, real_t crk4 )
{	// Linear combination Ax + s.kx and saves the result in aux_x
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;	
	
	if( idx < NX and idy < NY){
		auxp[IDX(idx,idy,idz)] = Pump[IDX(idx,idy,idz)] + kp[IDX(idx,idy,idz)] * crk4 ;
		auxs[IDX(idx,idy,idz)] = Signal[IDX(idx,idy,idz)] + ks[IDX(idx,idy,idz)] * crk4 ;
	}
	
	return ;
	
}


class Solver
{	// Difine the class Solver for modelling the fields propagation and heating
public:	
	complex_t *k1p, *k2p, *k3p, *k4p, *k1s, *k2s, *k3s, *k4s;
	complex_t *auxp, *auxs;	
	
	__host__ __device__ Solver()
	{	// Constructor
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
	}
	
	
	__host__ __device__ ~Solver()
	{	// Destructor
		cudaFree(k1p);	cudaFree(k2p);	cudaFree(k3p);		cudaFree(k4p);
		cudaFree(k1s);	cudaFree(k2s);	cudaFree(k3s);		cudaFree(k4s);	
		cudaFree(auxs);		cudaFree(auxp);		
	}
	
	void diffraction( Efields *A );
	void solverRK4( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, real_t z, uint slice );
	void SSFM( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, real_t z, uint slice );
	void CWES( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, bool save_only_last );
	void run( real_t Power, real_t waist, real_t focalpoint, real_t Temp, real_t T_inf, real_t Tpeltier, bool save_only_last, bool save_temperature );
};


void Solver::diffraction ( Efields *A )
{	// Applies the diffraction term to the electric fields
	// Parameters for kernels 2D
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
	
	// Set plan for cuFFT 2D//
	cufftHandle plan2D;	cufftPlan2d(&plan2D, NX, NY, CUFFT_C2C );
	
	cufftExecC2C(plan2D, (complex_t *)A->Pump, (complex_t *)A->PumpQ, CUFFT_FORWARD);
	CHECK(cudaDeviceSynchronize());
	cufftExecC2C(plan2D, (complex_t *)A->Signal, (complex_t *)A->SignalQ, CUFFT_FORWARD);
	CHECK(cudaDeviceSynchronize());	
	
	kernelComplexProduct<<<grid2D, block2D>>> (A->AuxQ, A->PropPump, A->PumpQ);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(A->PumpQ, A->AuxQ, nBytes2Dc, cudaMemcpyDeviceToDevice));
	
	kernelComplexProduct<<<grid2D, block2D>>> (A->AuxQ, A->PropSignal, A->SignalQ);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(A->SignalQ, A->AuxQ, nBytes2Dc, cudaMemcpyDeviceToDevice));
	
	cufftExecC2C(plan2D, (complex_t *)A->PumpQ, (complex_t *)A->Pump, CUFFT_INVERSE);
	CHECK(cudaDeviceSynchronize());
	cuFFTscale<<<grid2D, block2D>>>(A->Pump);
	CHECK(cudaDeviceSynchronize());		
	cufftExecC2C(plan2D, (complex_t *)A->SignalQ, (complex_t *)A->Signal, CUFFT_INVERSE);
	CHECK(cudaDeviceSynchronize());
	cuFFTscale<<<grid2D, block2D>>>(A->Signal);
	CHECK(cudaDeviceSynchronize());
	
	cufftDestroy(plan2D);
	
	return ;
}


void Solver::solverRK4( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, real_t z, uint slice )
{	// Applies the Fourh-order Runge-Kutta Method with fixed step size dz
	// This function apply the fourth-order Runge-Kutta method	
	
	// Parameters for kernels 2D
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
	real_t *z_device;	cudaMalloc((void**)&z_device, sizeof(real_t));
	cudaMemcpy(z_device, &z, sizeof(real_t), cudaMemcpyHostToDevice);
	
	#ifdef THERMAL
	uint *slice_device;	cudaMalloc((void**)&slice_device, sizeof(uint));
	cudaMemcpy(slice_device, &slice, sizeof(uint), cudaMemcpyHostToDevice);
	
	DK->setDKInt0();
	DK->IntegrateDK( T, slice );
	
	real_t increment = 0.0;
	dAdz<<<grid2D,block2D>>>( this->k1p, this->k1s, A->Pump, A->Signal, T->Tfinal, DK->DKint, 0.5*increment, slice_device );
	CHECK(cudaDeviceSynchronize()); 
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k1p, this->k1s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	increment = 0.5;
	dAdz<<<grid2D,block2D>>>( this->k2p, this->k2s, this->auxp, this->auxs, T->Tfinal, DK->DKint, 0.5*increment, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k2p, this->k2s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	dAdz<<<grid2D,block2D>>>( this->k3p, this->k3s, this->auxp, this->auxs, T->Tfinal, DK->DKint, 0.5*increment, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k3p, this->k3s, 1.0 );
	CHECK(cudaDeviceSynchronize());   
	increment = 1.0;
	dAdz<<<grid2D,block2D>>>( this->k4p, this->k4s, this->auxp, this->auxs, T->Tfinal, DK->DKint, 0.5*increment, slice_device );
	CHECK(cudaDeviceSynchronize());
	
	kernelRK4<<<grid2D,block2D>>>( A->Pump, A->Signal, this->k1p, this->k1s, this->k2p, this->k2s, this->k3p, this->k3s,this->k4p, this->k4s );
	CHECK(cudaDeviceSynchronize());
	cudaFree((void *) slice_device);
	
	
	#else
	real_t *Temp_device;	cudaMalloc((void**)&Temp_device, sizeof(real_t));
	cudaMemcpy(Temp_device, &Temp, sizeof(real_t), cudaMemcpyHostToDevice);
	
	real_t increment = 0.0;
	dAdz<<<grid2D,block2D>>>( this->k1p, this->k1s, A->Pump, A->Signal, DK->DK, Temp_device, z_device, 0.5*increment );
	CHECK(cudaDeviceSynchronize()); 
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k1p, this->k1s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	increment = 0.5;
	dAdz<<<grid2D,block2D>>>( this->k2p, this->k2s, this->auxp, this->auxs, DK->DK, Temp_device, z_device, 0.5*increment );
	CHECK(cudaDeviceSynchronize());
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k2p, this->k2s, 0.5 );
	CHECK(cudaDeviceSynchronize());   
	dAdz<<<grid2D,block2D>>>( this->k3p, this->k3s, this->auxp, this->auxs, DK->DK, Temp_device, z_device, 0.5*increment );
	CHECK(cudaDeviceSynchronize());
	
	linealCombination<<<grid2D,block2D>>>( this->auxp, this->auxs, A->Pump, A->Signal, this->k3p, this->k3s, 1.0 );
	CHECK(cudaDeviceSynchronize());   
	increment = 1.0;
	dAdz<<<grid2D,block2D>>>( this->k4p, this->k4s, this->auxp, this->auxs, DK->DK, Temp_device, z_device, 0.5*increment );
	CHECK(cudaDeviceSynchronize());
	
	kernelRK4<<<grid2D,block2D>>>( A->Pump, A->Signal, this->k1p, this->k1s, this->k2p, this->k2s, this->k3p, this->k3s,this->k4p, this->k4s );
	CHECK(cudaDeviceSynchronize());
	
	cudaFree((void *) Temp_device);
	#endif
	cudaFree((void *) z_device); 
	return ;
	
}


void Solver::SSFM( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, real_t z, uint slice )
{	// Applies the Split-Step Fourier Method using scheme N/2 - L - N/2
	solverRK4 ( A, T, DK, Temp, z, slice ); // Runge-Kutta 4 for dz/2
	diffraction ( A ) ; // diffraction for dz
	solverRK4 ( A, T, DK, Temp, z, slice ); // Runge-Kutta 4 for dz/2
	return ;
}


void Solver::CWES( Efields *A, Tfield *T, PhaseMatching *DK, real_t Temp, bool save_only_last )
{	// Solve the coupled-wave equations along the nonlinear crystal	
	real_t z = 0.0;
	uint s = 0;
	while( s < NZ) {
		std::cout << "Completed " << s*100/NZ << "%" << "\t\r" << std::flush;
		if(save_only_last){// save fields in the last slice (save_only_last = true)
			if( s == NZ-1 ){
				// 				std::cout << "Saving only last slice" << std::endl;
				saveMatrixComplexGPU ( A->Pump, "Pump_out" );
				saveMatrixComplexGPU ( A->Signal, "Signal_out" );
			}
		}
		else{
			if( s <= NZ-1 ){// save fields in every slice (save_only_last = false)
				// 				std::cout << "Saving slice #" << s << std::endl;
				saveMatrixComplexGPU ( A->Pump, "Pump_"+std::to_string(s) );
				saveMatrixComplexGPU ( A->Signal, "Signal_"+std::to_string(s) );
			}				
		}
		// Split-Step Fourier Method (SSFM) for the z-th step
		SSFM( A, T,  DK, Temp, z, s );
		
		#ifdef THERMAL
		T->upDateQ( A,  s );
		#endif
		
		z+=dz; 
		s++;// next slice
	}
	
	return ;		
}


void Solver::run( real_t Power, real_t waist, real_t focalpoint, real_t Temp, real_t T_inf, real_t Tpeltier, bool save_only_last, bool save_temperature )
{	// Run the solver in the body of the main function.
	
	// 	Class instances
	Efields *A				= new Efields;
	Tfield *T				= new Tfield;
	PhaseMatching *DK	= new PhaseMatching;
	
	T->setTemperature( Temp );	// Set inicial conditions for temperature field
	
	#ifdef THERMAL
	T->setBottomOvens( Tpeltier); // set oven temperature	
	T->setInitialQ();	// Set initial Q=0
	
	uint global_count = 0, num_of_iter = 20;	// accounts for the global iterations
	
	while (global_count < num_of_iter )
	{
		std::cout << "\n\nGlobal iteration #" << global_count << "\n\n" << std::endl;
		std::cout << "Solving 3D Heat Equation in the steady-state...\n" << std::endl;
		
		real_t tol = 5e-4/TSIZE; // tolerance for convergence
		std::cout << "\nTemperature tolerance = " << tol << "\n" << std::endl;
		
		real_t Reduced_sum = 1.0, aux_sum ;	// compares 2 consecutive steps in thermal calculations
		uint counter_T = 0; // accounts for the thermal iterations 
		std::cout.precision(20);
		
		while ( Reduced_sum >= tol )
		{
			T->upDate(T_inf); // update temperature
			uint chkconv = 5000; 
			if (counter_T % chkconv == 0)
			{	// Check temperature convergence every chkconv iterations
				if (counter_T > 0) aux_sum = Reduced_sum;
				Reduced_sum = T->checkConvergence();
				std::cout << "\u03A3|Tf-Ti|²/N³ at #" << counter_T << " iteration is: " << Reduced_sum << std::endl;
				// check if reduced_sum reaches constant value
				if ( Reduced_sum==aux_sum ) Reduced_sum = 0.5*tol;
				// saveTensorRealGPU ( T->Tfinal, "Tfinal" );
			}
			CHECK(cudaMemcpy(T->Tinic, T->Tfinal, nBytes3Dr, cudaMemcpyDeviceToDevice));
			
			counter_T++;            
		}
		std::cout << counter_T << " iterations -> steady-state." << "\t\r" << std::flush;
		
		// Set inicial conditions for electric fields
		DK->setDKFromTemperature( T );	// For simulations with thermal calculations
		A->setInputPump( Power, focalpoint, waist,  T->Tfinal );
		A->setNoisyField();
		A->setPropagators( T->Tfinal );
		
		std::cout << "\n\nSolving Coupled-Wave Equations (CWEs)...\n" << std::endl;
		this->CWES( A, T, DK, Temp, save_only_last );
		
		if (counter_T == 1 and global_count != 0) global_count = num_of_iter;
		
		global_count++;
	}
		
	if(save_temperature)	
		saveTensorRealGPU ( T->Tfinal, "Tfinal" );
	
	#else
	// Set inicial conditions for electric fields
	A->setInputPump( Power, focalpoint, waist, T->Tfinal );
	A->setNoisyField();
	A->setPropagators( T->Tfinal );
	
	std::cout << "\n\nEvolution in the crystal...\n" << std::endl;
	DK->setInicialDKConstant( Temp );	// for simulations without thermal calculations	
	this->CWES( A, T, DK, Temp, save_only_last );
	#endif
	
	std::cout << "Deallocating memory" << std::endl;
	delete A, T, DK;
	
	return ;
}

#endif // -> #ifdef _SOLVERCUH
