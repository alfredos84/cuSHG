/*---------------------------------------------------------------------------*/
// * This file contains the class Tfield to set the temperature in the same
// * grid than class Efields.
/*---------------------------------------------------------------------------*/


#ifndef _TFIELDCUH
#define _TFIELDCUH

#pragma once


__global__ void kernelDifSquareVectors( real_t *Dif, real_t *Tfinal, real_t *Tinic )
{	// Computes |Tfinal-Tinic|² and is used in reduced sum.
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy < NY) ){ // and (idz < NZ) ){
			Dif[IDX(idx,idy,idz)] = (Tfinal[IDX(idx,idy,idz)] - Tinic[IDX(idx,idy,idz)])*(Tfinal[IDX(idx,idy,idz)] - Tinic[IDX(idx,idy,idz)]);
		}
	}
	
	return;
}


__global__ void kernelSetTemperature(real_t *Tinic, real_t *Temp)
{	// Set the temperature of the nonlinear cristal at T = Temp that usually is the
	// phase-matching temperature. However, this is false in focused beams.
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy < NY) ){
			Tinic[IDX(idx,idy,idz)] = *Temp;
		}
	}
	
	return;
	
}


__global__ void kernelSetBottomOvensTemperature( real_t *Tfield, real_t *TPeltier )
{	// Set the bottom face temperature/s. 
	// For a single oven set Tpeltier = Tpeltier2
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = 0;

	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idz < NZ) ){
			Tfield[IDX(idx,idy,idz)] = *TPeltier;
		}
	}
	
	return;
	
}


__global__ void kernelSetOvenSurrounded( real_t *Tfield, real_t *TPeltier )
{	// Set the bottom, top and lateral faces temperature/s. 
	// For a single oven set Tpeltier = Tpeltier2
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	real_t nz = real_t(NZ);
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy == 0) and (idz < nz) ){
			Tfield[IDX(idx,idy,idz)] = *TPeltier;
		}
		if ( (idx < NX) and (idy == NY-1) and (idz < nz) ){
			Tfield[IDX(idx,idy,idz)] = *TPeltier;
		}
		if ( (idy < NY) and (idx == 0) and (idz < nz) ){
			Tfield[IDX(idx,idy,idz)] = *TPeltier;
		}
		if ( (idy < NY) and (idx == NX-1) and (idz < nz) ){
			Tfield[IDX(idx,idy,idz)] = *TPeltier;
		}
	}
	
	return;
	
}


__global__ void kernelHeatEquationTopOpen (real_t *Tf, real_t *Ti, real_t *Tenvironment, real_t *Q )
{	// Change if other boundary conditions are required, Currently, the bottom face is in contact with
	// two ovens (Peltier cells) set in setPeltiers() class method.	
	real_t T_inf = *Tenvironment;
	// grid steps
	
	real_t Bix = heat_trf_coeff*dx/thermal_cond, Biy = heat_trf_coeff*dy/thermal_cond, Biz = heat_trf_coeff*dz/thermal_cond;
	real_t ax = dx*dx/(2*dy*dy), ay = dy*dy/(2*dx*dx), az = dz*dz/(2*dx*dx); // for faces points
	real_t bx = dx*dx/(2*dz*dz), by = dy*dy/(2*dz*dz), bz = dz*dz/(2*dy*dy); // for faces points
	real_t av = dz/(2*dx), bv = dx/(2*dz), cv = dx*dz/(4*dy*dy), V = 1/(av+bv+2*cv+0.5*(Bix+Biz)); // for vertical sides
	real_t ah = dz/(2*dy), bh = dy/(2*dz), ch = dy*dz/(4*dx*dx), H = 1/(ah+bh+2*ch+0.5*(Biy+Biz)); // for horizontal sides
	real_t ad = dx/(2*dy), bd = dy/(2*dx), cd = dx*dy/(4*dz*dz), D = 1/(ad+bd+2*cd+0.5*(Bix+Biy)); // for depth sides
	real_t AV = dy*dz/dx, BV = dx*dz/dy, CV = dx*dy/dz, BiD = Bix*dy+Biy*dz+Biz*dx, GV = 1/(AV+BV+CV+BiD); //for vertices 
	real_t L2 = 1/(2/(dx*dx) + 2/(dy*dy) + 2/(dz*dz)); // for inner points
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		//* Inner points (x,y,z = 1 to N-2 ) */
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = L2 * ( (Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)])/(dx*dx) + (Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)])/(dy*dy) + (Ti[IDX(idx,idy,idz+1)] + Ti[IDX(idx,idy,idz-1)])/(dz*dz) + Q[IDX(idx,idy,idz)]/thermal_cond );
		}		
		
		// FACES /
		// Left  (x = 0)
		if ( (idx == 0) and (idy < NY-1) and (idy > 0) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx+1,idy,idz)] + ax*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + bx*(Ti[IDX(idx,idy,idz+1)] + Ti[IDX(idx,idy,idz-1)]) + Bix*T_inf ) / (1 + 2*ax + 2*bx + Bix) ;
		}
		// Right (x = Lx)
		if ( (idx == NX-1) and (idy < NY-1) and (idy > 0) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx-1,idy,idz)] + ax*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + bx*(Ti[IDX(idx,idy,idz+1)] + Ti[IDX(idx,idy,idz-1)]) + Bix*T_inf ) / (1 + 2*ax + 2*bx + Bix) ;
		}	
		// Top   (y = Ly)
		if ( (idx < NX-1) and (idx > 0) and (idy == NY-1) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx,idy-1,idz)] + ay*(Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)]) + by*(Ti[IDX(idx,idy,idz+1)] + Ti[IDX(idx,idy,idz-1)]) + Biy*T_inf ) / (1 + 2*ay + 2*by + Biy) ;
		}	
		// Back   (z = 0)
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx,idy,idz+1)] + az*(Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)]) + bz*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + Biz*T_inf ) / (1 + 2*az + 2*bz + Biz) ;
		}
		// Front  (z = Lcr)
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx,idy,idz-1)] + az*(Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)]) + bz*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + Biz*T_inf ) / (1 + 2*az + 2*bz + Biz) ;
		}
		
		
		// SIDES /	
		// 	VERTICAL (x=0, x=Lx, z=0, z=Lcr, y libre)
		// (x=0, z=0)
		if ( (idx == 0) and (idy < NY-1) and (idy > 0) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] =  (av*Ti[IDX(idx+1,idy,idz)] + bv*Ti[IDX(idx,idy,idz+1)] + cv*(Ti[IDX(idx,idy+1,idz)]+Ti[IDX(idx,idy-1,idz)]) + 0.5*T_inf*(Bix+Biz))*V ;
		}
		// (x=Lx, z=0)
		if ( (idx == NX-1) and (idy < NY-1) and (idy > 0) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] =  (av*Ti[IDX(idx-1,idy,idz)] + bv*Ti[IDX(idx,idy,idz+1)] + cv*(Ti[IDX(idx,idy+1,idz)]+Ti[IDX(idx,idy-1,idz)]) + 0.5*T_inf*(Bix+Biz))*V ;
		}
		// (x=Lx, z=Lcr)
		if ( (idx == NX-1) and (idy < NY-1) and (idy > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] =  (av*Ti[IDX(idx-1,idy,idz)] + bv*Ti[IDX(idx,idy,idz-1)] + cv*(Ti[IDX(idx,idy+1,idz)]+Ti[IDX(idx,idy-1,idz)]) + 0.5*T_inf*(Bix+Biz))*V ;
		}
		// (x=0, z=Lcr)
		if ( (idx == 0) and (idy < NY-1) and (idy > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] =  (av*Ti[IDX(idx+1,idy,idz)] + bv*Ti[IDX(idx,idy,idz-1)] + cv*(Ti[IDX(idx,idy+1,idz)]+Ti[IDX(idx,idy-1,idz)]) + 0.5*T_inf*(Bix+Biz))*V ;
		}
		
		// HORIZONTAL (y=0, y=Ly, z=0, z=Lcr)
		// (y=Ly, z=0)
		if ( (idy == NY-1) and (idx < NX-1) and (idx > 0) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] = (ah*Ti[IDX(idx,idy-1,idz)] + bh*Ti[IDX(idx,idy,idz+1)] + ch*(Ti[IDX(idx-1,idy,idz)]+Ti[IDX(idx+1,idy,idz)]) + 0.5*T_inf*(Biy+Biz))*H ;
		}
		// (y=Ly, z=Lcr)
		if ( (idy == NY-1) and (idx < NX-1) and (idx > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] = (ah*Ti[IDX(idx,idy-1,idz)] + bh*Ti[IDX(idx,idy,idz-1)] + ch*(Ti[IDX(idx-1,idy,idz)]+Ti[IDX(idx+1,idy,idz)]) + 0.5*T_inf*(Biy+Biz))*H ;
		}
		
		// DEPTH (x=0, x=Lx, y=0, y=Ly)
		// (x=0, y=Ly)
		if ( (idx == 0) and (idy == NY-1) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = (ad*Ti[IDX(idx,idy-1,idz)] + bd*Ti[IDX(idx+1,idy,idz)] + cd*(Ti[IDX(idx,idy,idz+1)]+Ti[IDX(idx,idy,idz-1)]) + 0.5*T_inf*(Bix+Biy))*D ;
		}
		// (x=Lx, y=Lx)
		if ( (idx == NX-1) and (idy == NY-1) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = (ad*Ti[IDX(idx,idy-1,idz)] + bd*Ti[IDX(idx-1,idy,idz)] + cd*(Ti[IDX(idx,idy,idz+1)]+Ti[IDX(idx,idy,idz-1)]) + 0.5*T_inf*(Bix+Biy))*D ;
		}
		
		// VERTICES /
		// (x = 0, y = Ly, z = 0)
		if ( (idx == 0) and (idy == NY-1) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] =  ( AV*Ti[IDX(idx+1,idy,idz)] + BV*Ti[IDX(idx,idy-1,idz)] + CV*Ti[IDX(idx,idy,idz+1)] + T_inf*BiD )*GV ;
		}
		// (x = 0, y = Ly, z = Lcr)
		if ( (idx == 0) and (idy == NY-1) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] =  ( AV*Ti[IDX(idx+1,idy,idz)] + BV*Ti[IDX(idx,idy-1,idz)] + CV*Ti[IDX(idx,idy,idz-1)] + T_inf*BiD )*GV ;
		}
		
		// (x = Lx, y = Ly, z = 0)
		if ( (idx == NX-1) and (idy == NY-1) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] =  ( AV*Ti[IDX(idx-1,idy,idz)] + BV*Ti[IDX(idx,idy-1,idz)] + CV*Ti[IDX(idx,idy,idz+1)] + T_inf*BiD )*GV ;
		}
		// 	// (x = Lx, y = Ly, z = Lcr)
		if ( (idx == NX-1) and (idy == NY-1) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] =  ( AV*Ti[IDX(idx-1,idy,idz)] + BV*Ti[IDX(idx,idy-1,idz)] + CV*Ti[IDX(idx,idy,idz-1)] + T_inf*BiD )*GV ;
		}
		// 
	}
	
	return ;
}



__global__ void kernelHeatEquationOvenSurrounded (real_t *Tf, real_t *Ti, real_t *Tenvironment, real_t *Q )
{	// Change if other boundary conditions are required, Currently, the bottom face is in contact with
	// two ovens (Peltier cells) set in setPeltiers() class method.	
	real_t T_inf = *Tenvironment;
	// grid steps
	
	real_t Biz = heat_trf_coeff*dz/thermal_cond;
	real_t az = dz*dz/(2*dx*dx); // for faces points
	real_t bz = dz*dz/(2*dy*dy); // for faces points
	
	real_t L2 = 1/(2/(dx*dx) + 2/(dy*dy) + 2/(dz*dz)); // for inner points
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		//* Inner points (x,y,z = 1 to N-2 ) */
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz < NZ-1) and (idz > 0) ){
			Tf[IDX(idx,idy,idz)] = L2 * ( (Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)])/(dx*dx) + (Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)])/(dy*dy) + (Ti[IDX(idx,idy,idz+1)] + Ti[IDX(idx,idy,idz-1)])/(dz*dz) + Q[IDX(idx,idy,idz)]/thermal_cond );
		}		
		
		// FACES
		
		// Back   (z = 0)
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz == 0) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx,idy,idz+1)] + az*(Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)]) + bz*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + Biz*T_inf ) / (1 + 2*az + 2*bz + Biz) ;
		}
		
		// Front  (z = Lcr)
		if ( (idx < NX-1) and (idx > 0) and (idy < NY-1) and (idy > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] = ( Ti[IDX(idx,idy,idz-1)] + az*(Ti[IDX(idx+1,idy,idz)] + Ti[IDX(idx-1,idy,idz)]) + bz*(Ti[IDX(idx,idy+1,idz)] + Ti[IDX(idx,idy-1,idz)]) + Biz*T_inf ) / (1 + 2*az + 2*bz + Biz) ;
		}
	}
	
	return ;
}


__global__ void kernelSetQ0( real_t *Q )
{	// Set the initial internal heat source
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy < NY) ){
			Q[IDX(idx,idy,idz)] = 0.0;
		}
	}
	
	return;
	
}


__global__ void kernelSetQ( real_t *Q, real_t *Tfinal, complex_t *Pump, complex_t *Signal, uint *slice )
{	// Set the internal heat source Q = αpIp +αsIs + βsIs²
	
	uint s = *slice;
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if ( (idx < NX) and (idy < NY) ){
		Q[IDX(idx,idy,s)] = alpha_crp*0.5*C*EPS0*n(lp, Tfinal[IDX(idx,idy,s)])*CpxAbs2(Pump[IDX(idx,idy,idz)]) + alpha_crs*0.5*C*EPS0*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)]) + beta_crs*powf(0.5*C*EPS0*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)]), 2);
	}
	
	
	return;
	
}


class Tfield
{	// Difine the class Tfield for calculation thermal profile
	
public:
	real_t *Tinic, *Tfinal, *Taux;
	real_t *Q;
	
	__device__ __host__ Tfield()
	{	// Constructor
		cudaMalloc((void **)&Tinic, nBytes3Dr );
		cudaMalloc((void **)&Tfinal, nBytes3Dr );
		cudaMalloc((void **)&Taux, nBytes3Dr );
		cudaMalloc((void **)&Q, nBytes3Dr );
	}
	
	__host__ __device__ ~Tfield()
	{	// Destructor
		cudaFree((void *)Tinic);cudaFree((void *)Tfinal);
		cudaFree((void *)Taux);	cudaFree((void *)Q);
	}
	
	void setTemperature( real_t Temp )
	{	// Set inicial temperature
		real_t *Temp_device;	cudaMalloc((void**)&Temp_device, sizeof(real_t));
		cudaMemcpy(Temp_device, &Temp, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelSetTemperature<<<grid3D, block3D>>>( this->Tinic, Temp_device );
		CHECK(cudaDeviceSynchronize());	
		kernelSetTemperature<<<grid3D, block3D>>>( this->Tfinal, Temp_device );
		CHECK(cudaDeviceSynchronize());			
		
		cudaFree((void *) Temp_device);
		
		return ;
	}
	
	void setBottomOvens( real_t Tpeltier )
	{	// Set ovens temperature in an open-top configuration
		
		real_t *Tpeltier_device;cudaMalloc((void**)&Tpeltier_device, sizeof(real_t));
		cudaMemcpy(Tpeltier_device, &Tpeltier, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelSetBottomOvensTemperature<<<grid3D, block3D>>>( this->Tinic, Tpeltier_device );
		CHECK(cudaDeviceSynchronize());	
		kernelSetBottomOvensTemperature<<<grid3D, block3D>>>( this->Tfinal, Tpeltier_device );
		CHECK(cudaDeviceSynchronize());
		
		cudaFree((void *) Tpeltier_device);
		
		return ;
	}
	
	void setOvenSurrounded( real_t Tpeltier )
	{	// Set ovens temperature in an surrounded-oven configuration
		
		real_t *Tpeltier_device;cudaMalloc((void**)&Tpeltier_device, sizeof(real_t));
		cudaMemcpy(Tpeltier_device, &Tpeltier, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelSetOvenSurrounded<<<grid3D, block3D>>>( this->Tinic, Tpeltier_device );
		CHECK(cudaDeviceSynchronize());	
		kernelSetOvenSurrounded<<<grid3D, block3D>>>( this->Tfinal, Tpeltier_device );
		CHECK(cudaDeviceSynchronize());
		
		cudaFree((void *) Tpeltier_device);
		
		return ;
	}
	
	void upDate(real_t T_inf)
	{	// Update the temperature by solving the heat equation
		
		real_t *T_inf_device;	cudaMalloc((void**)&T_inf_device, sizeof(real_t));
		cudaMemcpy(T_inf_device, &T_inf, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelHeatEquationTopOpen<<<grid3D, block3D>>>( this->Tfinal, this->Tinic, T_inf_device, this->Q );
		CHECK(cudaDeviceSynchronize());
		
		cudaFree ((void *)T_inf_device);
		
		return ;
		
	}
	
	real_t checkConvergence(void)
	{	// Check whether the temperature changes during iterations
		real_t Reduced_sum;
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		thrust::device_ptr<real_t> dDif_ptr(this->Taux);
		kernelDifSquareVectors<<<grid3D, block3D>>> ( this->Taux, this->Tfinal, this->Tinic );
		CHECK(cudaDeviceSynchronize());
		Reduced_sum = sqrt(thrust::reduce(dDif_ptr, dDif_ptr + TSIZE ))/TSIZE;
		return Reduced_sum;
	}
	
	void setInitialQ(void)
	{	// Set the inicial  internal heat source Q=0
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelSetQ0<<<grid3D, block3D>>>( this->Q );
		CHECK(cudaDeviceSynchronize());	
		
		return ;
		
	}
	
	void upDateQ( Efields *A, uint s )
	{	// Update the internal heat source Q
		uint *ss_device;	cudaMalloc((void**)&ss_device, sizeof(uint));
		cudaMemcpy(ss_device, &s, sizeof(uint), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		kernelSetQ<<<grid3D, block3D>>>( this->Q, this->Tfinal, A->Pump, A->Signal, ss_device);
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) ss_device);
		
		return ;
		
	}
	
};



#endif // -> #ifdef _TFIELDCUH
