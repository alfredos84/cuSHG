/*---------------------------------------------------------------------------*/
// * This file contains the class PhaseMatching.
/*---------------------------------------------------------------------------*/


#ifndef _PHASEMATCHINGCUH
#define _PHASEMATCHINGCUH

#pragma once


__global__ void kernelSetDK0( real_t *DKint )
{	// Set mismatch matrix filled of constant values
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	if ( (idx < NX) and (idy < NY) )
	{
		DKint[IDX(idx,idy,0)] = 0.0;
	}
	
	return ;	
}


__global__ void kernelIntegrateDK(real_t *DKint, real_t *DK,  real_t *Tfinal, uint *slice)
{	// This kernel integrates the DK matrix from z'=0 to z'=z
	
	uint s = *slice; 
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz <= s; idz++ )
	{
		if ( (idx < NX) and (idy < NY) )
		{
			DKint[IDX(idx,idy,0)] += DK[IDX(idx,idy,idz)];
		}
	}
	return ;
	
}


__global__ void kernelSetInicialDKThermal( real_t *DK, real_t *Tfinal )
{	// Set mismatch matrix for thermal calculations	
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	real_t alpha = 2.2e-6, beta = -5.9e-9;	 // thermal expansion coefficients
	
	for (uint idz = 0; idz < NZ; idz++)
	{
		if ( (idx < NX) and (idy < NY) )
		{
			DK[IDX(idx,idy,idz)] = 2*PI*( 2*n(lp, Tfinal[IDX(idx,idy,idz)])/lp - n(ls, Tfinal[IDX(idx,idy,idz)])/ls + 1/(Lambda*(1. + alpha*(Tfinal[IDX(idx,idy,idz)]-25.) + beta*powf(Tfinal[IDX(idx,idy,idz)]-25.,2))) );			
		}
	}
	
	return ;
	
}


__global__ void kernelSetInicialDKConstant( real_t *DK, real_t *Temperature )
{	// Set mismatch matrix filled of constant values	
	real_t Temp = *Temperature;
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	real_t alpha = 2.2e-6, beta = -5.9e-9; // thermal expansion coefficients
	
	for (uint idz = 0; idz < NZ; idz++)
	{
		if ( (idx < NX) and (idy < NY) )
		{
			DK[IDX(idx,idy,idz)] = 2*PI*( 2*n(lp, Temp)/lp - n(ls, Temp)/ls + 1/(Lambda*(1. + alpha*(Temp-25.) + beta*powf(Temp-25.,2))) );
// 			DK[IDX(idx,idy,idz)] = 3.2/LZ;
		}
	}
	return ;	
}


class PhaseMatching
{	// Difine the class PhaseMatching for calculations on the mismatch factor
public:
	real_t *DK, *DKint;
	
	__host__ __device__ PhaseMatching()
	{	// Constructor
		cudaMalloc((void **)&DK, nBytes3Dr ); 
		cudaMalloc((void **)&DKint, nBytes2Dr ); 
	}
	
	__host__ __device__ ~PhaseMatching()
	{	// Destructor
		cudaFree((void *)DK);
		cudaFree((void *)DKint);
	}
	
	void setDKInt0( void )
	{	// Set the DKint matrix = 0
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);	dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		kernelSetDK0<<<grid2D, block2D>>>( this->DKint );
		CHECK(cudaDeviceSynchronize());	
		
		return ;
	}
	
	void IntegrateDK( Tfield *T, uint slice )
	{	// Sum in each slice the accumulated mismatch factor
		uint *slice_device;	cudaMalloc((void**)&slice_device, sizeof(uint));
		cudaMemcpy(slice_device, &slice, sizeof(uint), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block2D(BLKX, BLKY);		dim3 grid2D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		kernelIntegrateDK<<<grid2D, block2D>>>( this->DKint, this->DK, T->Tfinal, slice_device );
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) slice_device);
		
		return ;
	}
	
	void setDKFromTemperature( Tfield *T )
	{	// Set the DK matrix for thermal calculations
		//* Parameters for kernels 2D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		kernelSetInicialDKThermal<<<grid3D, block3D>>>( this->DK, T->Tfinal );
		CHECK(cudaDeviceSynchronize());	
		return ;
	}	
	
	void setInicialDKConstant( real_t Temp )
	{	// Set the DK matrix as a constant 
		real_t *Temp_device;	cudaMalloc((void**)&Temp_device, sizeof(real_t));
		cudaMemcpy(Temp_device, &Temp, sizeof(real_t), cudaMemcpyHostToDevice);
		
		//* Parameters for kernels 2D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
		kernelSetInicialDKConstant<<<grid3D, block3D>>>( this->DK, Temp_device );
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) Temp_device);
		
		return ;
	}
	
};

#endif // -> #ifdef _PHASEMATCHINGCUH
