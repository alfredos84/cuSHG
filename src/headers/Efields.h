/*---------------------------------------------------------------------------*/
// * This file contains the class Efields to set all the electric fields involved in
// * the simulations.
/*---------------------------------------------------------------------------*/


#ifndef _EFIELDSCUH
#define _EFIELDSCUH

#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions for FFT */

__global__ void cuFFTscale(complex_t *A)
{	// Scales a vector after Fourier transforms (CUFFT_INVERSE mode)	
	
	real_t size = NX*NY;
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		A[IDX(idx,idy,idz)] = A[IDX(idx,idy,idz)] / size;
	}
	
	return ;
	
}


__global__ void fftShift2DH( complex_t *Field, complex_t *aux)
{	 // Swap horizontally the values un a matrix
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	uint c = (int) floor((real_t)NX/2);
	
	if (idx < c and idy < NY){
		Field[IDX(idx,idy,0)+c]  =  aux[IDX(idx,idy,0)];
// 		Field[IDX(idx,idy,0)+c].y  =  aux[IDX(idx,idy,0)].y;
		Field[IDX(idx,idy,0)]  =  aux[IDX(idx,idy,0)+c];
// 		Field[IDX(idx,idy,0)].y  =  aux[IDX(idx,idy,0)+c].y;
	}
	
	return ;
}


__global__ void fftShift2DV( complex_t *Field, complex_t *aux)
{	// Swap vertically the values un a matrix
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	uint r = (int) floor((real_t)NY/2);
	
	if (idy < r and idx < NX){
		Field[(idy+r)*NX+idx]  =  aux[IDX(idx,idy,0)];
// 		Field[(idy+r)*NX+idx].y  =  aux[IDX(idx,idy,0)].y;
		Field[IDX(idx,idy,0)]  =  aux[(idy+r)*NX+idx];
// 		Field[IDX(idx,idy,0)].y  =  aux[(idy+r)*NX+idx].y;
	}  
	
	return ;
}


void fftShift2D ( complex_t* d_trans )
{	// Standard fftshift in 2D
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
	complex_t *aux;	CHECK(cudaMalloc((void **)&aux, nBytes2Dc));
	CHECK(cudaMemcpy(aux, d_trans, nBytes2Dc, cudaMemcpyDeviceToDevice));
	fftShift2DV<<<grid2D, block2D>>>(d_trans, aux);
	cudaDeviceSynchronize();
	CHECK(cudaMemcpy(aux, d_trans, nBytes2Dc, cudaMemcpyDeviceToDevice));
	fftShift2DH<<<grid2D, block2D>>>(d_trans, aux);
	cudaDeviceSynchronize();
	CHECK(cudaFree(aux));
	
	return ;	
}


__global__ void setPump( complex_t *Pump, real_t *pump_power, real_t *f, real_t *w0, real_t *Tfinal )
{	// Set initial pump as a Gaussian beam
	
	real_t Power = *pump_power;	real_t focalpoint = *f;
	real_t waistX = *w0;	real_t Temp = Tfinal[IDX(NX/2,NY/2,0)];
	
	real_t PI	  = 3.14159265358979323846;     // pi
	complex_t Im  = make_cuComplex(0.0f, 1.0f); // imaginary number
	real_t uX     = 0.5*NX;
	real_t uY     = 0.5*NY;
	real_t waistY = waistX;
	real_t wX2    = waistX*waistX;
	real_t wY2    = waistY*waistY;
	real_t zRX    = PI*n(lp,Temp)*wX2/lp;
	real_t zRY    = PI*n(lp,Temp)*wY2/lp;
	real_t Ap0    = sqrtf(4*Power/(EPS0*C*PI*n(lp, Temp)*wX2));
	real_t etaX   = focalpoint/zRX;
	real_t etaY   = focalpoint/zRY;
	complex_t MX  = (1-Im*etaX);
	complex_t MY  = (1-Im*etaY);
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		Pump[IDX(idx,idy,idz)] = (Ap0/MX) * CpxExp( ((-powf((idx-uX)*dx,2) - powf((idy-uY)*dy,2)) / (wX2*MX)) );
	}
	
	// 	if( idx < NX and idy < NY){
	// 		Pump[IDX(idx,idy,idz)] = (Ap0/(CpxSqrt(MX)*CpxSqrt(MY))) * CpxExp( (-powf((idx-uX)*dx,2))/(wX2*MX) - powf((idy-uY)*dy,2)/(wY2*MY) );
	// 	}
	
	return;
	
}


__global__ void beamPropagator ( complex_t *eiQz_pump, complex_t *eiQz_signal, real_t *Tfinal )
{	// This kernel set the beam propagator operators useful in diffraction calculations
	real_t Temp = Tfinal[IDX(NX/2,NY/2,0)];
	real_t PI	= 3.14159265358979323846;     // pi
	real_t kp	= 2*PI*n(lp, Temp)/lp;
	real_t ks	= 2*PI*n(ls, Temp)/ls;
	real_t dfX= 1/dx/NX;	real_t dfY  = 1/dy/NY;
	real_t uX	= 0.5*NX;	real_t uY   = 0.5*NY;
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		eiQz_pump[IDX(idx,idy,idz)] = CpxExp(-dz*(2*powf(PI,2)/kp * ( dfX*dfX*powf(idx - uX,2) + dfY*dfY*powf(idy - uY,2)) + 2*PI*dfX*tanf(rhop)*(idx-uX)) ); 
		eiQz_signal[IDX(idx,idy,idz)] = CpxExp(-dz*(2*powf(PI,2)/ks * ( dfX*dfX*powf(idx - uX,2) + dfY*dfY*powf(idy - uY,2))  + 2*PI*dfX*tanf(rhos)*(idx-uX)) ); 
	}
	
	return ;	
}


void noiseGeneratorCPU ( complex_t *A )
{	// Noise generator for initial signal/idler vectors 
	uint seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<real_t> distribution(0.0,1.0e-20);
	
	real_t nsx, nsy;    
	for (int i=0; i<NX*NY; ++i) {
		nsx = distribution(generator); A[i].x = nsx;
		nsy = distribution(generator); A[i].y = nsy;
	}
	
	return ;	
}


void noiseGeneratorGPU ( complex_t *A_gpu )
{	// Noise generator for initial signal/idler vectors in GPU
	complex_t *A = (complex_t*)malloc(nBytes2Dc);
	noiseGeneratorCPU ( A );
	CHECK(cudaMemcpy( A_gpu, A, nBytes2Dc, cudaMemcpyHostToDevice));
	free(A);
	
	return ;
}


class Efields
{	
public:
	complex_t *Pump, *Signal;
	complex_t *PumpQ, *SignalQ;
	complex_t *PropPump, *PropSignal;		
	complex_t *AuxQ;
	
	__device__ __host__ Efields()
	{	// Constructor
		cudaMalloc((void **)&Pump, nBytes2Dc );
		cudaMalloc((void **)&Signal, nBytes2Dc );
		cudaMalloc((void **)&PumpQ, nBytes2Dc );
		cudaMalloc((void **)&SignalQ, nBytes2Dc );
		cudaMalloc((void **)&PropPump, nBytes2Dc );
		cudaMalloc((void **)&PropSignal, nBytes2Dc );
		cudaMalloc((void **)&AuxQ, nBytes2Dc );		
	}
	
	__device__ __host__ ~Efields()
	{	// Destructor
		cudaFree((void *)Pump);		cudaFree((void *)Signal);
		cudaFree((void *)PumpQ);	cudaFree((void *)SignalQ);
		cudaFree((void *)PropPump);	cudaFree((void *)PropSignal);
		cudaFree((void *)AuxQ);
	}
	

	void setInputPump( real_t Power, real_t focalpoint, real_t waistX, real_t *Tfinal )
	{	// Set initial pump with a given power, beam waist and focal position
		real_t *Power_device;		cudaMalloc((void**)&Power_device, sizeof(real_t));
		cudaMemcpy(Power_device, &Power, sizeof(real_t), cudaMemcpyHostToDevice);		
		real_t *focalpoint_device;		cudaMalloc((void**)&focalpoint_device, sizeof(real_t));
		cudaMemcpy(focalpoint_device, &focalpoint, sizeof(real_t), cudaMemcpyHostToDevice);
		real_t *waistX_device;		cudaMalloc((void**)&waistX_device, sizeof(real_t));
		cudaMemcpy(waistX_device, &waistX, sizeof(real_t), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		setPump<<<grid2D, block2D>>>( this->Pump, Power_device, focalpoint_device, waistX_device, Tfinal );
		cudaDeviceSynchronize();
		
		cudaFree(Power_device);
		cudaFree(focalpoint_device);
		cudaFree(waistX_device);
		
		// 		std::cout << "Setting Pump field" << std::endl;
		return ;
	}
	
	void setNoisyField()
	{	// Set signal vector as a noisy input
		noiseGeneratorGPU ( this->Signal );
		// 		std::cout << "Setting Signal field" << std::endl;
		return ;
	}	


	void setPropagators( real_t *Tfinal )
	{	// Set vectors for beam propagators
		
		// Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		// 		* Diffraction operation ∇²_{xy}: Beam propagator for Pump and Signal
		beamPropagator<<<grid2D, block2D>>> ( this->PropPump, this->PropSignal, Tfinal );
		CHECK(cudaDeviceSynchronize());
		
		fftShift2D ( this->PropPump );	fftShift2D ( this->PropSignal );

		return ;
	}
	
};


#endif // -> #ifdef _EFIELDSCUH
