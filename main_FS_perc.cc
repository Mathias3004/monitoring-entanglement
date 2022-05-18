//
//  main.cpp
//  firing squad: sampling trajectories of decaying spins with 2x2 Haar random unitaries in layered pattern 
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
    
    cout << endl << "running p=" << p << " for N=" << N_spin << " on " << env.nnodes() << " cores\n\n";
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",300);
    auto type = input.getString("type","localization");
    auto loc_type = input.getString("loc_type","pseudo");
    
    auto verbose = input.getInt("verbose",0); 
    auto save = input.getInt("save",1) == 1; 

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //-----------------------------------------------------------
    // START
    
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    auto sites = SpinHalf(N_spin, args);
    
    // for saving
    string files = output_folder + "/dat_N_" + to_string(N_spin) + "_p_" + to_string(p);
    
    // Max Renyi entropy to evaluate
    int alpha_max = 4;
    


    //-----------------------------------------------------------------------
    // START TRAJECTORY
    //-----------------------------------------------------------------------
    
    auto start = clock();
    
    // Sample a unitary according to protocol type
    MatrixXcf U;
    if (type=="percolation"){
        // random offset 0 or 1, where to start first block
        int offset = rand() % 2;
        
        // The sampled unitary
        U = random_U_percolation(N_spin, p, offset);
    }
    else if (type=="localization"){
        cout << loc_type << endl;
        int D = 4*N_spin;
        U = U_BS_layer(N_spin, D, p, loc_type); 
    }

     // vector with jumps from U
    vector<MPO> C;
    add_decay_U(C, sites, U);

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
        
        // evaluate Renyi entropies entropies
        vector<vector<double>> Sv(alpha_max);
        for (int i_site=0; i_site<=psi.N(); i_site++){
            for (int i=0; i<alpha_max; i++){
                Sv[i].push_back( S_vn(psi,i_site, i+1) );
            }
        }
        
        if (verbose >= 2){
            cout << "S_VN: ";
            for (int i=0; i<(int)Sv[0].size(); i++){
                cout << Sv[0][i] << " ";
            }
            cout << endl;
            print_occupations(psi, args);
             
        }

        if (verbose >= 1){
            cout << "rank " << env.rank() << ": state " << i_jump+1 << "/" << N_spin << " saved in " << 
            (clock() - start_jump)/ (double) CLOCKS_PER_SEC << "s, MBD=" << maxM(psi) << endl;
            env.barrier();
            if (env.rank()==0){
                cout << endl<< endl;
            }
        }
        
        
        env.barrier();
        // save that shit whilst keeping order of nodes
        if (save){
            for (int n=0; n<env.nnodes(); n++){
                if (n==env.rank()){
                    for (int i=0; i<alpha_max; i++){
                        string fn = files + "_S_" + to_string(i+1) + "_";
                        save_entropies(Sv[i],fn,i_jump);
                        
                    }
                }
                env.barrier();
            }
        
        // wait for all cores to finish
        env.barrier();
        }
    }
    
    // wait for all processes to finish
    env.barrier();
    if (env.rank()==0){
        cout << endl << "Run finished in " << (clock() - start)/ (double) CLOCKS_PER_SEC << "s. " << env.nnodes() << " trajectories evaluated." << endl << endl;
        cout << "parameters: " << "p=" << p << ", N=" << N_spin << endl << endl;
        
    }
        
        
    
    
    return 0;
    
}