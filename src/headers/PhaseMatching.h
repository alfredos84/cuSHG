/*---------------------------------------------------------------------------*/
// * This file contains the class PhaseMatching.
/*---------------------------------------------------------------------------*/


#ifndef _PHASEMATCHINGCUH
#define _PHASEMATCHINGCUH

#pragma once


// This kernel updates the DK matrix by accumulating the phase along the propagation
__global__ void KernelAccumulatedDK(real_t *DK, real_t *Tfinal, uint *slice)
{	
	
	uint s = *slice; 
		
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if ( (idx < NX) and (idy < NY) ){
		DK[IDX(idx,idy,idz)] = DK[IDX(idx,idy,idz)] + ( dz*2*PI*(2*n(lp, Tfinal[IDX(idx,idy,s)])/lp - n(ls, Tfinal[IDX(idx,idy,s)])/ls + 1/(Lambda )) );
	}
	
	return ;
	
}

// Set mismatch matrix for thermal calculations
__global__ void KernelSetInicialDKThermal( real_t *DK, real_t *Tinic )
{	
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if ( (idx < NX) and (idy < NY) ){
		DK[IDX(idx,idy,idz)] = 2*PI*( 2*n(lp, Tinic[IDX(idx,idy,idz)])/lp - n(ls, Tinic[IDX(idx,idy,idz)])/ls + 1/(Lambda) );
	}
	
	return ;
	
}


// Set mismatch matrix filled of constant values 
__global__ void KernelSetInicialDKConstant( real_t *DK, real_t *Temperature )
{	
	real_t Tpm = *Temperature;
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if ( (idx < NX) and (idy < NY) ){
		DK[IDX(idx,idy,idz)] = 2.1/LZ; //2*PI*( 2*n(lp, Tpm)/lp - n(ls, Tpm)/ls + 1/(Lambda) );
	}
	
	return ;	
}


class PhaseMatching
{
public:
	real_t *DK;
	
	// Constructor
	__host__ __device__ PhaseMatching() {
		cudaMalloc((void **)&DK, nBytes2Dr ); 
		real_t *DK_cpu = new real_t[NX*NY];
		cudaMemcpy(DK, DK_cpu, nBytes2Dr, cudaMemcpyHostToDevice );
		delete[] DK_cpu;
	}
	
	// Destructor
	__host__ __device__ ~PhaseMatching() {
		cudaFree((void *)DK);
	}
	
	void SetAccumulatedDK( Tfield *T, uint slice )
	{
		uint *slice_device;	cudaMalloc((void**)&slice_device, sizeof(uint));
		cudaMemcpy(slice_device, &slice, sizeof(uint), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);		dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		KernelAccumulatedDK<<<grid2D, block2D>>>( this->DK, T->Tfinal, slice_device );
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) slice_device);
		
		return ;
	}
	
	void SetInicialDKThermal( Tfield *T )
	{
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);		dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		KernelSetInicialDKThermal<<<grid2D, block2D>>>( this->DK, T->Tinic );
		CHECK(cudaDeviceSynchronize());	
		return ;
	}	
	
	void SetInicialDKConstant( real_t Tpm )
	{
		real_t *Tpm_device;	cudaMalloc((void**)&Tpm_device, sizeof(real_t));
		cudaMemcpy(Tpm_device, &Tpm, sizeof(real_t), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);		dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		KernelSetInicialDKConstant<<<grid2D, block2D>>>( this->DK, Tpm_device );
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) Tpm_device);
		
		return ;
	}
	
};

#endif // -> #ifdef _PHASEMATCHINGCUH
