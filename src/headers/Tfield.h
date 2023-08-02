/*---------------------------------------------------------------------------*/
// * This file contains the class Tfield to set the temperature in the same
// * grid than class Efields.
/*---------------------------------------------------------------------------*/


#ifndef _TFIELDCUH
#define _TFIELDCUH

#pragma once

// Used in reduced sum
__global__ void KernelDifSquareVectors( real_t *Dif, real_t *Tfinal, real_t *Tinic )
{
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy < NY) ){ // and (idz < NZ) ){
			Dif[IDX(idx,idy,idz)] = (Tfinal[IDX(idx,idy,idz)] - Tinic[IDX(idx,idy,idz)])*(Tfinal[IDX(idx,idy,idz)] - Tinic[IDX(idx,idy,idz)]);
		}
	}
	
	return;
}


__global__ void KernelSetTemperature(real_t *Tinic, real_t *Tpm)
{
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idy < NY) ){
			Tinic[IDX(idx,idy,idz)] = *Tpm;
		}
	}
	
	return;
	
}


__global__ void KernelSetBottomOvensTemperature( real_t *Tinic, real_t *TPeltier1, real_t *TPeltier2 )
{
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = 0;
	
	real_t nz = real_t(NZ);
	
	for (uint idz = 0; idz < NZ; idz++){
		if ( (idx < NX) and (idz < nz/2.0) ){
			Tinic[IDX(idx,idy,idz)] = *TPeltier1;
		}
		if ( (idx < NX) and (idz >= nz/2.0) and (idz < NZ) ){
			Tinic[IDX(idx,idy,idz)] = *TPeltier2;
		}
	}
	
	return;
	
}

// Cambiar en caso de poner condiciones de contorno diferentes. Actualmente se encuentra con una T=Toven fija en el borde inferior-
__global__ void KernelHeatEquationBottomOven (real_t *Tf, real_t *Ti, real_t *Tenvironment, real_t *Q )
{	
	
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
			Tf[IDX(idx,idy,idz)] = (ah*Ti[IDX(idx,idy-1,idz)] + bh*Ti[IDX(idx,idy,idz+1)] + ch*(Ti[IDX(idx+1,idy,idz)]+Ti[IDX(idx+1,idy,idz)]) + 0.5*T_inf*(Biy+Biz))*H ;
		}
		// (y=Ly, z=Lcr)
		if ( (idy == NY-1) and (idx < NX-1) and (idx > 0) and (idz == NZ-1) ){
			Tf[IDX(idx,idy,idz)] = (ah*Ti[IDX(idx,idy-1,idz)] + bh*Ti[IDX(idx,idy,idz-1)] + ch*(Ti[IDX(idx+1,idy,idz)]+Ti[IDX(idx+1,idy,idz)]) + 0.5*T_inf*(Biy+Biz))*H ;
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

// Set the internal heat source
__global__ void KernelSetQ( real_t *Q, real_t *Tfinal, complex_t *Pump, complex_t *Signal, uint *slice )
{
	
	uint s = *slice;
	
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint idy = threadIdx.y + blockDim.y*blockIdx.y;
	uint idz = 0;
	
	if ( (idx < NX) and (idy < NY) ){
		Q[IDX(idx,idy,s)] = alpha_crp*0.5*C*EPS0*n(lp, Tfinal[IDX(idx,idy,s)])*CpxAbs2(Pump[IDX(idx,idy,idz)]) + alpha_crs*0.5*C*EPS0*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)]) + beta_crs*(0.5*C*EPS0*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)])) * (0.5*C*EPS0*n(ls,Tfinal[IDX(idx,idy,s)])*CpxAbs2(Signal[IDX(idx,idy,idz)]));
	}
	
	
	return;
	
}


class Tfield
{
public:
	real_t *Tinic, *Tfinal, *Taux;
	real_t *Q;
	
	// Constructor
	__device__ __host__ Tfield() {
		cudaMalloc((void **)&Tinic, nBytes3Dr ); real_t *Tinic_cpu = new real_t[TSIZE];
		cudaMalloc((void **)&Tfinal, nBytes3Dr );real_t *Tfinal_cpu = new real_t[TSIZE];
		cudaMalloc((void **)&Taux, nBytes3Dr );	real_t *Taux_cpu = new real_t[TSIZE];
		cudaMalloc((void **)&Q, nBytes3Dr ); real_t *Q_cpu = new real_t[TSIZE];
		cudaMemset(Q, 0, nBytes3Dr );
		
		cudaMemcpy(Tinic, Tinic_cpu, nBytes3Dr, cudaMemcpyHostToDevice );
		cudaMemcpy(Tfinal, Tfinal_cpu, nBytes3Dr, cudaMemcpyHostToDevice );
		cudaMemcpy(Taux, Taux_cpu, nBytes3Dr, cudaMemcpyHostToDevice );
		cudaMemcpy(Q, Q_cpu, nBytes3Dr, cudaMemcpyHostToDevice );
		
		delete[] Tinic_cpu;	delete[] Tfinal_cpu; 
		delete[] Taux_cpu;	delete[] Q_cpu;
	}
	
	// Destructor
	__host__ __device__ ~Tfield() {
		cudaFree((void *)Tinic);cudaFree((void *)Tfinal);
		cudaFree((void *)Taux);	cudaFree((void *)Q);
	}
	
	void SetTemperature( real_t Tpm )
	{
		real_t *Tpm_device;	cudaMalloc((void**)&Tpm_device, sizeof(real_t));
		cudaMemcpy(Tpm_device, &Tpm, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		KernelSetTemperature<<<grid3D, block3D>>>( this->Tinic, Tpm_device );
		CHECK(cudaDeviceSynchronize());	
		KernelSetTemperature<<<grid3D, block3D>>>( this->Tfinal, Tpm_device );
		CHECK(cudaDeviceSynchronize());			
		cudaFree((void *) Tpm_device);
		
		return ;
	}
	
	void SetPeltiers( real_t Tpeltier1, real_t Tpeltier2 )
	{
		real_t *Tpeltier1_device;cudaMalloc((void**)&Tpeltier1_device, sizeof(real_t));
		cudaMemcpy(Tpeltier1_device, &Tpeltier1, sizeof(real_t), cudaMemcpyHostToDevice);
		real_t *Tpeltier2_device;cudaMalloc((void**)&Tpeltier2_device, sizeof(real_t));
		cudaMemcpy(Tpeltier2_device, &Tpeltier2, sizeof(real_t), cudaMemcpyHostToDevice);		
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		KernelSetBottomOvensTemperature<<<grid3D, block3D>>>( this->Tinic, Tpeltier1_device, Tpeltier2_device );
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) Tpeltier1_device);
		cudaFree((void *) Tpeltier2_device);
		
		return ;
	}
	
	void UpDate(real_t T_inf)
	{
		real_t *T_inf_device;	cudaMalloc((void**)&T_inf_device, sizeof(real_t));
		cudaMemcpy(T_inf_device, &T_inf, sizeof(real_t), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		KernelHeatEquationBottomOven<<<grid3D, block3D>>>( this->Tfinal, this->Tinic, T_inf_device, this->Q );
		CHECK(cudaDeviceSynchronize());
		
		cudaFree ((void *)T_inf_device);
		
		return ;
		
	}
	
	real_t CheckConvergence(void)
	{
		real_t Reduced_sum;
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);		dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		thrust::device_ptr<real_t> dDif_ptr(this->Taux);
		KernelDifSquareVectors<<<grid3D, block3D>>> ( this->Taux, this->Tfinal, this->Tinic );
		CHECK(cudaDeviceSynchronize());
		Reduced_sum = sqrt(thrust::reduce(dDif_ptr, dDif_ptr + TSIZE ))/TSIZE;
		return Reduced_sum;
	}
	
	void UpDateQ( Efields *E, uint s )
	{
		uint *ss_device;	cudaMalloc((void**)&ss_device, sizeof(uint));
		cudaMemcpy(ss_device, &s, sizeof(uint), cudaMemcpyHostToDevice);
		//* Parameters for kernels 3D
		dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
		KernelSetQ<<<grid3D, block3D>>>( this->Q, this->Tfinal, E->Pump, E->Signal, ss_device);
		CHECK(cudaDeviceSynchronize());	
		
		cudaFree((void *) ss_device);
		
		return ;
	}
	
};



#endif // -> #ifdef _TFIELDCUH
