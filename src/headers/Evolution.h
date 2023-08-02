/*---------------------------------------------------------------------------*/
// * This file contains functions to solve the Split-Step Fourier method (SSMF)
// * needed to calculate the electric fields evolution through the nonlinear crystal.
// * 
// * In particular, this file should be used when only two equation describes the 
// * problem, i.e., parametric down-convertion or second-harmonic generation.
// * Only two frequencies are involved in theses problems.
/*---------------------------------------------------------------------------*/


#ifndef _EVOLUTIONCUH
#define _EVOLUTIONCUH

#pragma once




void EvolutionInCrystal( real_t Power, real_t waist, real_t focalpoint, real_t Tpm, real_t T_inf, real_t Tpeltier1, real_t Tpeltier2, bool save_only_last, bool save_temperature )
{
	// 	Class instances
	Efields *E = new Efields;
	Tfield *T = new Tfield;
	PhaseMatching *DK = new PhaseMatching;
	Solver *Sv = new Solver;
	
	// Set inicial conditions for temperatrature field
	T->SetTemperature( Tpm );
	T->SetPeltiers( Tpeltier1, Tpeltier2 );
// 	SaveTensorRealGPU ( T->Tinic, "Tinic" );
	
	#ifdef THERMAL
	DK->SetInicialDKThermal( T );	// for simulations with thermal calculations	
	
	thrust::device_ptr<real_t> dDif_ptr(T->Taux);
	uint global_count = 0;
	uint counter;						// accounts for the thermal iterations
	uint num_of_iter      = 20;			// accounts for the global iterations
	real_t Reduced_sum;	// compares 2 consecutive steps in thermal calculations
	real_t tol       =  1e-7; 			// tolerance for convergence	
	
	dim3 block3D(BLKX, BLKY);	dim3 grid3D((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY );
	while (global_count < num_of_iter ){
		std::cout << "\n\nGlobal iteration #" << global_count << "\n\n" << std::endl;		
		std::cout << "Solving 3D Heat Equation in the steady-state...\n" << std::endl;
		Reduced_sum = 1;	counter = 0; std::cout.precision(4);
		while ( Reduced_sum >= tol ){
			T->UpDate(T_inf);			
			KernelDifSquareVectors<<<grid3D, block3D>>> ( T->Taux, T->Tfinal, T->Tinic );
			CHECK(cudaDeviceSynchronize());
			Reduced_sum = sqrt(thrust::reduce(dDif_ptr, dDif_ptr + TSIZE ))/TSIZE;
// 			Reduced_sum = T->CheckConvergence();
			
			if (counter%5000==0)
				std::cout << "\u03A3|Tf-Ti|²/N³ at #" << counter << " iteration is: " << Reduced_sum << std::endl;
			
			CHECK(cudaMemcpy(T->Tinic, T->Tfinal, nBytes3Dr, cudaMemcpyDeviceToDevice));
			counter++;            
		}
		std::cout << counter << " iterations -> steady-state." << std::endl;
		
		// Set inicial conditions for electric fields
		E->SetInputPump( Power, focalpoint, waist, Tpm );
		E->SetNoisyField();
		E->SetPropagators( Tpm );
		
		std::cout << "\n\nSolving Coupled-Wave Equations (CWEs)...\n" << std::endl;
		Sv->CWES( E, T, DK, Tpm, save_only_last );
		
		if (counter == 1 and global_count != 0) global_count = num_of_iter;
		
		global_count++;
	}
	
	if(save_temperature)	SaveTensorRealGPU ( T->Tfinal, "Tfinal" );
	
	#else
	// Set inicial conditions for electric fields
	E->SetInputPump( Power, focalpoint, waist, Tpm );
	E->SetNoisyField();
	E->SetPropagators( Tpm );
	
	std::cout << "\n\nEvolution in the crystal...\n" << std::endl;
	DK->SetInicialDKConstant( Tpm );	// for simulations without thermal calculations	
	Sv->CWES( E, T, DK, Tpm, save_only_last );
	#endif
	
	std::cout << "Deallocating memory" << std::endl;
	delete E, T, DK, Sv;
	
	return ;
}


#endif // -> #ifdef _EVOLUTIONCUH

