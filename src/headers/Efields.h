/*---------------------------------------------------------------------------*/
// * This file contains the class Efields to set all the electric fields involved in
// * the simulations.
/*---------------------------------------------------------------------------*/


#ifndef _EFIELDSCUH
#define _EFIELDSCUH

#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions for FFT */
/** Scales a vector after Fourier transforms (CUFFT_INVERSE mode) */
__global__ void CUFFTscale(complex_t *A)
{
	
	real_t size = NX*NY;
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		A[IDX(idx,idy,idz)] = A[IDX(idx,idy,idz)] / size;
	}
	
	return ;
	
}


__global__ void FFTShift2DH( complex_t *Field, complex_t *aux)
{
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	uint c = (int) floor((real_t)NX/2);
	
	if (idx < c and idy < NY){
		Field[IDX(idx,idy,0)+c].x  =  aux[IDX(idx,idy,0)].x;
		Field[IDX(idx,idy,0)+c].y  =  aux[IDX(idx,idy,0)].y;
		Field[IDX(idx,idy,0)].x  =  aux[IDX(idx,idy,0)+c].x;
		Field[IDX(idx,idy,0)].y  =  aux[IDX(idx,idy,0)+c].y;
	}
	
	return ;
}


__global__ void FFTShift2DV( complex_t *Field, complex_t *aux)
{
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	uint r = (int) floor((real_t)NY/2);
	
	if (idy < r and idx < NX){
		Field[(idy+r)*NX+idx].x  =  aux[IDX(idx,idy,0)].x;
		Field[(idy+r)*NX+idx].y  =  aux[IDX(idx,idy,0)].y;
		Field[IDX(idx,idy,0)].x  =  aux[(idy+r)*NX+idx].x;
		Field[IDX(idx,idy,0)].y  =  aux[(idy+r)*NX+idx].y;
	}  
	
	return ;
}


void FFTShift2D ( complex_t* d_trans )
{
	dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
	complex_t *aux;	CHECK(cudaMalloc((void **)&aux, nBytes2Dc));
	CHECK(cudaMemcpy(aux, d_trans, nBytes2Dc, cudaMemcpyDeviceToDevice));
	FFTShift2DV<<<grid2D, block2D>>>(d_trans, aux);
	cudaDeviceSynchronize();
	CHECK(cudaMemcpy(aux, d_trans, nBytes2Dc, cudaMemcpyDeviceToDevice));
	FFTShift2DH<<<grid2D, block2D>>>(d_trans, aux);
	cudaDeviceSynchronize();
	CHECK(cudaFree(aux));
	
	return ;	
}


__global__ void SetPump( complex_t *Pump, real_t *pump_power, real_t *f, real_t *w0, real_t *Temperature )
{
	
	real_t Power = *pump_power;	real_t focalpoint = *f;
	real_t waistX = *w0;	real_t Tpm = *Temperature;
	
	real_t PI	  = 3.14159265358979323846;     // pi
	complex_t Im  = make_cuComplex(0.0f, 1.0f); // imaginary number
	real_t uX     = 0.5*NX;
	real_t uY     = 0.5*NY;
	real_t waistY = waistX;
	real_t wX2    = waistX*waistX;
	real_t wY2    = waistY*waistY;
	real_t zRX    = PI*n(lp,Tpm)*wX2/lp;
	real_t zRY    = PI*n(lp,Tpm)*wY2/lp;
	real_t Ap0    = sqrtf(Power/(EPS0*C*PI*n(lp, Tpm)*wX2));
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


__global__ void BeamPropagator ( complex_t *eiQz_pump, complex_t *eiQz_signal, real_t *Temperature )
{	
	real_t Tpm = *Temperature;
	real_t PI	= 3.14159265358979323846;     // pi
	real_t kp   = 2*PI*n(lp, Tpm)/lp;
	real_t ks   = 2*PI*n(ls, Tpm)/ls;
	real_t dfX  = 1/dx/NX;	real_t dfY  = 1/dy/NY;
	real_t uX   = 0.5*NX;	real_t uY   = 0.5*NY;
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if( idx < NX and idy < NY){
		eiQz_pump[IDX(idx,idy,idz)] = CpxExp(-dz*(2*powf(PI,2)/kp * ( dfX*dfX*powf(idx - uX,2) + dfY*dfY*powf(idy - uY,2)) + 2*PI*dfX*tanf(rho)*(idx-uX) - 0.5*alpha_crp) ); 
		eiQz_signal[IDX(idx,idy,idz)] = CpxExp(-dz*(2*powf(PI,2)/ks * ( dfX*dfX*powf(idx - uX,2) + dfY*dfY*powf(idy - uY,2))  + 2*PI*dfX*tanf(rho)*(idx-uX) - 0.5*alpha_crs) ); 
	}
	
	return ;	
}


// Noise generator for initial signal/idler vectors 
void NoiseGeneratorCPU ( complex_t *A )
{	
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


void NoiseGeneratorGPU ( complex_t *A_gpu )
{
	complex_t *A = (complex_t*)malloc(nBytes2Dc);
	NoiseGeneratorCPU ( A );
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
	
	// Constructor
	__device__ __host__ Efields( ) {
		cudaMalloc((void **)&Pump, nBytes2Dc );	complex_t *Pump_cpu = new complex_t[NX*NY];
		cudaMalloc((void **)&Signal, nBytes2Dc );	complex_t *Signal_cpu = new complex_t [NX*NY];	
		cudaMalloc((void **)&PumpQ, nBytes2Dc );	complex_t *PumpQ_cpu = new complex_t[NX*NY];
		cudaMalloc((void **)&SignalQ, nBytes2Dc );	complex_t *SignalQ_cpu = new complex_t[NX*NY];
		cudaMalloc((void **)&PropPump, nBytes2Dc );	complex_t *PropPump_cpu = new complex_t[NX*NY];
		cudaMalloc((void **)&PropSignal, nBytes2Dc );	complex_t *PropSignal_cpu = new complex_t[NX*NY];
		cudaMalloc((void **)&AuxQ, nBytes2Dc );	complex_t *AuxQ_cpu = new complex_t[NX*NY];
		cudaMemcpy(Pump, Pump_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(Signal, Signal_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(PumpQ, PumpQ_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(SignalQ, SignalQ_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(PropPump, PropPump_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(PropSignal, PropSignal_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		cudaMemcpy(AuxQ, AuxQ_cpu, nBytes2Dc, cudaMemcpyHostToDevice );
		
		delete[] Pump_cpu;	delete[] Signal_cpu;  
		delete[] PumpQ_cpu;	delete[] SignalQ_cpu;  
		delete[] PropPump_cpu;	delete[] PropSignal_cpu;
		delete[] AuxQ_cpu;		
	}
	
	// Destructor
	__device__ __host__ ~Efields() {
		cudaFree((void *)Pump);		cudaFree((void *)Signal);
		cudaFree((void *)PumpQ);	cudaFree((void *)SignalQ);
		cudaFree((void *)PropPump);	cudaFree((void *)PropSignal);
		cudaFree((void *)AuxQ);
	};
	
	void SetInputPump( real_t Power, real_t focalpoint, real_t waistX, real_t Tpm )
	{		
		real_t *Power_device;		cudaMalloc((void**)&Power_device, sizeof(real_t));
		cudaMemcpy(Power_device, &Power, sizeof(real_t), cudaMemcpyHostToDevice);		
		real_t *focalpoint_device;		cudaMalloc((void**)&focalpoint_device, sizeof(real_t));
		cudaMemcpy(focalpoint_device, &focalpoint, sizeof(real_t), cudaMemcpyHostToDevice);
		real_t *waistX_device;		cudaMalloc((void**)&waistX_device, sizeof(real_t));
		cudaMemcpy(waistX_device, &waistX, sizeof(real_t), cudaMemcpyHostToDevice);
		real_t *Tpm_device;	cudaMalloc((void**)&Tpm_device, sizeof(real_t));
		cudaMemcpy(Tpm_device, &Tpm, sizeof(real_t), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		SetPump<<<grid2D, block2D>>>( this->Pump, Power_device, focalpoint_device, waistX_device, Tpm_device );
		cudaDeviceSynchronize();
		
		cudaFree(Power_device);	cudaFree(focalpoint_device);
		cudaFree(waistX_device);	cudaFree(Tpm_device);
		
// 		std::cout << "Setting Pump field" << std::endl;
		return ;
	}
	
	void SetNoisyField()
	{
		NoiseGeneratorGPU ( this->Signal );
// 		std::cout << "Setting Signal field" << std::endl;
		return ;
	}	
	
	void SetPropagators( real_t Tpm )
	{
		real_t *Tpm_device;
		cudaMalloc((void**)&Tpm_device, sizeof(real_t));
		cudaMemcpy(Tpm_device, &Tpm, sizeof(real_t), cudaMemcpyHostToDevice);
		
		// Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);		dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		// 		* Diffraction operation ∇²_{xy}: Beam propagator for Pump and Signal
		BeamPropagator<<<grid2D, block2D>>> ( this->PropPump, this->PropSignal, Tpm_device );
		CHECK(cudaDeviceSynchronize());
		
		FFTShift2D ( this->PropPump );	FFTShift2D ( this->PropSignal );
		cudaFree(Tpm_device);
		return ;
	}
	
};


#endif // -> #ifdef _EFIELDSCUH
