//
//  main.cpp
//  sampling trajectories of decaying spins with Haar random unitaries
//
//  Created by Mathias on 7/8/2020
//

#include <stdio.h>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include "itensor/all.h"
#include "itensor/util/parallel.h"

#include "random_matrix.h"
#include "entropy.h"
#include "decay.h"
#include "functions.h"


using namespace std;
using namespace itensor;
using namespace Eigen;

int main ( int argc, char *argv[] )
{
        //---------------------------------------------------------
    //----------------------initialize MPI---------------------
    Environment env(argc,argv);
    
    //---------------------------------------------------------
    //---------parse parameters from input file ---------------
    auto input = InputGroup(argv[1],"input");
    
    // command line inputs
    auto p = atof(argv[2]);
    auto N_spin = atoi(argv[3]);
    auto N_sample = atoi(argv[4]);
    
    cout << endl << "running p=" << p << " for N=" << N_spin << " with " << N_sample << " samples on " << env.nnodes() << " cores\n\n";
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",200);
    
    auto dt = input.getReal("dt", 0.01);
    auto t_ev = input.getReal("t_ev",1.);
    auto N_collect = input.getInt("N_collect",(int) 50);
    auto N_layer = input.getInt("N_layer", N_spin);
    
    auto gamma = input.getReal("gamma",1.);
    auto Omega = input.getReal("Omega",10.);
    auto type = input.getString("type","BS");
    auto verbose = input.getInt("verbose",0) == 1; 

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //-----------------------------------------------------------
    // START
    
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    auto sites = SpinHalf(N_spin, args);
    
    // for saving
    string files = output_folder + "/dat_N_" + to_string(N_spin) + "_p_" + to_string(p);
    
    // The Hamiltonian and TE
    auto H = H_drive(sites, Omega, 0., gamma);
    auto U_ev = toExpH<ITensor>(H,Cplx_i*dt);
    
    // operator N for obtaining probabilities
    auto ampo_N = AutoMPO(sites);
    for (int i=1; i<N_spin; i++){
        ampo_N += 1., "projUp",i;
    }
    
    auto mpo_N = MPO(ampo_N);
    
    // time steps each collection
    int N_step = (int) (t_ev/(double) N_collect/dt);
    
    // loop over number of trajectories for each core
    for (int is = 0; is<N_sample; is++){
        
        auto start = clock();
        
        // initial state
        auto state = InitState(sites);
        for(int i = 1; i <= N_spin; ++i)
        {
            state.set(i,"Dn");
        }
        auto psi = MPS(state);
        
        // time evolve
        for (int ic=0; ic<N_collect; ic++){
            
            auto start_step = clock();
            int count_jump = 0;
            
            for (int is=0; is<N_step; is++){
                
                // probability to jump
                double pj = real(overlapC(psi,mpo_N,psi))*dt*gamma;
                bool click = pj > ((double) rand() / (RAND_MAX));
                
                if (click){
                    
                    // the jumps from BS brick pattern layers with 2x2 Haar U, each layer is saved

                    // random offset 0 or 1, where to start first 2x2 block
                    int offset = rand() % 2;
                    vector<MatrixXcf> v_U = random_U_haar_layer_vector(N_spin, N_layer, offset);
                    
                    
                    // select which layer and add jumps to vector C
                    double coeff = p;
                    double pl = ((double) rand() / (RAND_MAX));
                    double sum_coeff = coeff;
                    
                    vector<MPO> C;
                    for (int i=0; i<=N_layer; i++){
                        if (i==N_layer){ coeff = pow(1.-p, N_layer);}
                        
                        if (sum_coeff > pl){
                            add_decay_U(C, sites, v_U[i]);
                            break;
                        }
                        
                        coeff *= (1.-p);
                        sum_coeff += coeff;
                    }
                    
                    
                    
                    // select jump and apply to psi
                    SelectAndApplyJump(psi, C, args);
                    // normalize
                    psi.position(1,args);
                    normalize(psi);
                    count_jump++;
                }
                
                // non-hermitian TE (every dt now)
                psi = exactApplyMPO(U_ev,psi,args);
                
                // normalize
                psi.position(1,args);
                normalize(psi);
            }
            
            
            if (verbose){
                cout << "rank " << env.rank() << ":  \t" << "Run " << is+1 << "/" << N_sample << "\tstate " << ic+1 << "/" << N_collect << " reached in " << 
                (clock() - start_step)/ (double) CLOCKS_PER_SEC << "s\tMBD=" << maxM(psi) << "\tPj=" << (double) (count_jump)/(double) (N_step) << endl;
                // print densities
                /*for (int j=1; j<=N_spin; j++){
                    psi.position(j);
            
                    ITensor ket = psi.A(j);
                    ITensor bra = dag(prime(ket,Site));
            
                    ITensor Szjop = sites.op("projUp",j);
        
                    //take an inner product 
                    auto szj = (bra*Szjop*ket).real();
                    cout << szj << " ";
                }*/
                

            }
            
            // save that shit (make sure it's ordered)
            
            save_entropies(psi,files,ic);
            env.barrier();
            /*for (int in=0; in<env.nnodes(); in++){
                if (env.rank()==in){ }
                env.barrier();
            }*/

            if (env.rank()==0){
               cout << "\nRun " << is+1 << "/" << N_sample << ", state " << ic+1 << "/" << N_collect << " reached in " << 
                (clock() - start_step)/ (double) CLOCKS_PER_SEC << "s, MBD=" << maxM(psi) << ", Pj=" << (double) (count_jump)/(double) (N_step) << endl << endl;
            }
        }

        
        // wait for all processes to finish
        env.barrier();
        if (env.rank()==0){
            cout << endl << "Run " << is+1 << " finished in " << (clock() - start)/ (double) CLOCKS_PER_SEC << "s" << endl << endl;
            
        }
        
    }
        
    
    
    return 0;
    
}