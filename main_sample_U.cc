//
//  main.cpp
//  sampling trajectories of decaying spins with Haar random unitaries, with probability p of direct detection
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
    
    auto detection = input.getString("detection","direct");
    auto N_layer = input.getInt("N_layer",N_spin);

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //-----------------------------------------------------------
    // START
    
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    auto sites = SpinHalf(N_spin, args);
    
    // for saving
    string files = output_folder + "/dat_N_" + to_string(N_spin) + "_p_" + to_string(p);
    
    // loop over number of samples per core, sample new U each time
    for (int is = 0; is<N_sample; is++){
        
        auto start = clock();
        
        // vector with possible decays
        vector<MPO> C;
        
        // either direct detection or NxN unitary mixing or layered 2x2 unitary
        // mixing, after each layer probability p of detection
        if (detection == "direct"){
            // sample Haar U and add decays, prob 1-p
            MatrixXcf U = random_U_haar(N_spin);
            add_decay_U(C, sites, U, sqrt(1.-p));
            
            // identity for direct measurements, prob p
            MatrixXcf Id = MatrixXcf::Identity(N_spin,N_spin);
            add_decay_U(C, sites, Id, sqrt(p));
        }
        else{
            // random offset 0 or 1, where to start first 2x2 block
            int offset = rand() % 2;
            vector<MatrixXcf> v_U = random_U_haar_layer_vector(N_spin, N_layer, offset);
            
            double coeff = p;
            for (int i=0; i<=N_layer; i++){
                if (i==N_layer){ coeff = pow(1.-p, N_layer);}
                add_decay_U(C, sites, v_U[i], sqrt(coeff));
                coeff *= (1.-p);
            }
            
        }
        
        auto N_channel = C.size();

        // initial state
        auto state = InitState(sites);
        for(int i = 1; i <= N_spin; ++i)
        {
            state.set(i,"Up");
        }
        auto psi = MPS(state);
        
        // loop over the number of jumps in the chain
        for (int i_jump = 0; i_jump<N_spin; i_jump++){
            
            auto start_jump = clock();
            
            vector<double> Pc_i;
            vector<MPS> save_psi;
            
            double sum_Pc = 0.;
            
            for (int ic = 0; ic<N_channel; ic++){
                MPS c_psi = applyMPO(C[ic],psi,args);
                double Pc = real(overlapC(c_psi, c_psi));
                
                sum_Pc += Pc;

                //store
                save_psi.push_back(c_psi);
                Pc_i.push_back(Pc);
            }
            
            // select jump and apply to psi
            double e = ((double) rand() / (RAND_MAX))*sum_Pc;
            double sum_pj = 0.;
            for (int ic=0; ic<N_channel; ic++){
                sum_pj += Pc_i[ic];
                if (sum_pj>e){
                    psi = save_psi[ic];
                    psi.position(1,args);
                    normalize(psi);
                    
                    // save that shit
                    save_entropies(psi,files,i_jump);
                    break;
                }
            }
            
            cout << "rank " << env.rank() << ": " << "run " << is+1 << "/" << N_sample << ", state " << i_jump+1 << "/" << N_spin << " saved in " << 
            (clock() - start_jump)/ (double) CLOCKS_PER_SEC << "s, MBD=" << maxM(psi) << endl;
            env.barrier();
            if (env.rank()==0){
                cout << endl<< endl;
            
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