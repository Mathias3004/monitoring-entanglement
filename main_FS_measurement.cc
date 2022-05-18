//
//  main.cpp
//  firing squad: sampling trajectories of decaying spins with Haar random unitaries, alternated with spin measurements
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
    
    auto type = input.getString("type","U");
    auto verbose = input.getInt("verbose",0) == 1; 

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
        
        // the jumps from BS brick pattern layers, either unitaries or maximal mxing
        vector<MPO> C;
        
        // Haar Unitary and create decays
        MatrixXcf U;
        U = random_U_haar(N_spin);
        add_decay_U(C, sites, U);
        
        /*if (verbose){
            cout << "The matrix U=\n" << U << endl;
            cout << "Unitary? Ud*U=\n" << U.adjoint()*U << endl;
        }*/
        

        // initial state: all spins up
        auto state = InitState(sites);
        for(int i = 1; i <= N_spin; ++i)
        {
            state.set(i,"Up");
        }
        auto psi = MPS(state);

        // loop over the number of jumps in the chain
        for (int i_jump = 0; i_jump<N_spin; i_jump++){
            
            auto start_jump = clock();
            
            // select which jump to apply and find new psi
            SelectAndApplyJump(psi, C, args);

            // select and apply spin measurements
            SelectAndApplyMeasurements(psi, p, args);

            // evaluate entropies
            vector<double> Sv;
            for (int i_site=0; i_site<=psi.N(); i_site++){
                Sv.push_back( S_vn(psi,i_site) );
            }

            if (verbose){
                cout << "rank " << env.rank() << ": " << "run " << is+1 << "/" << N_sample << ", state " << i_jump+1 << "/" << N_spin << " saved in " << 
                (clock() - start_jump)/ (double) CLOCKS_PER_SEC << "s, MBD=" << maxM(psi) << endl;
                
                print_occupations(psi,args);

                env.barrier();
                if (env.rank()==0){
                    cout << endl<< endl;
                }
            }
            
            env.barrier();
            // save that shit whilst keeping order of nodes
            for (int n=0; n<env.nnodes(); n++){
                if (n==env.rank()){
                    save_entropies(Sv,files,i_jump);
                }
                env.barrier();
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