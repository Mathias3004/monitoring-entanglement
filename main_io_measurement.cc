//
//  main.cpp
//  sampling trajectories of \sigma^- and \sigma^+ with random unitaries composed of one layer 2x2 Haar gates. 
//  After each jump there is probability p for the spins to be measured.
//
//  Created by Mathias on 24/5/2021
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
    
    cout << endl << "running measurement p=" << p << " for N=" << N_spin << " with " << N_sample << " samples on " << env.nnodes() << " cores\n\n";
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",200);
    
    auto dt = input.getReal("dt", 0.01);
    auto t_ev = input.getReal("t_ev",1.);
    auto N_collect = input.getInt("N_collect",(int) 50);
    
    auto gamma_decay = input.getReal("gamma_decay",1.);
    auto gamma_absorb = input.getReal("gamma_absorb", gamma_decay);
    auto N_layer = input.getInt("N_layer",1);
    
    auto Omega = input.getReal("Omega",0.);
    auto verbose = input.getInt("verbose",0) == 1; 

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //-----------------------------------------------------------
    // START
    
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    auto sites = SpinHalf(N_spin, args);
    
    // for saving
    string files = output_folder + "/dat_N_" + to_string(N_spin) + "_p_" + to_string(p);
    
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
        
        // <N> excitations start
        int N = (int) real(overlapC(psi,mpo_N,psi));
        
        // time evolve
        for (int ic=0; ic<N_collect; ic++){
            
            auto start_step = clock();
            int count_jump_decay = 0;
            int count_jump_absorb = 0;
            
            for (int is=0; is<N_step; is++){
                
                 // probability decay
                double pj_decay = N*gamma_decay*dt;
                bool click_decay = pj_decay > ((double) rand() / (RAND_MAX));
                
                if (click_decay){
                    // the jumps from BS brick pattern layers with 2x2 Haar U, select layer and apply
                    NLayerApplyJump(psi, "S-", N_layer, args);
                    
                    // count decay jumps and particle number
                    count_jump_decay++;
                    N--;
                }
                
                // probability absorb
                double pj_absorb = (N_spin-N)*gamma_absorb*dt;
                bool click_absorb = pj_absorb > ((double) rand() / (RAND_MAX));
                
                if (click_absorb){
                    // the jumps from BS brick pattern layers with 2x2 Haar U, select layer and apply
                    NLayerApplyJump(psi, "S+", N_layer, args);
                    
                    count_jump_absorb++;
                    N++;
                }
                
                // perform measurements with probability rate p
                if (p>1E-4){
                    SelectAndApplyMeasurements(psi, p*dt, args);
                }

            }
            
            // evaluate entropies
            vector<double> Sv;
            for (int i_site=0; i_site<=psi.N(); i_site++){
                Sv.push_back( S_vn(psi,i_site) );
            }
            
            if (verbose){
                cout << "rank " << env.rank() << ":  \t" << "Run " << is+1 << "/" << N_sample << "\tstate " << ic+1 << "/" << N_collect << " reached in " << 
                (clock() - start_step)/ (double) CLOCKS_PER_SEC << "s\tMBD=" << maxM(psi) << "\tP_decay=" << (double) (count_jump_decay)/(double) (N_step) << 
                "\tP_absorb=" << (double) (count_jump_absorb)/(double) (N_step) << endl;
            }
            
            // wait for all cores to finish
            env.barrier();
            
            // save that shit whilst keeping order of nodes
            for (int n=0; n<env.nnodes(); n++){
                if (n==env.rank()){
                    save_entropies(Sv,files,ic);
                    //print_occupations(psi, args);
                }
                env.barrier();
            }
            
            // wait for all cores to finish
            env.barrier();


            if (env.rank()==0 ){
                cout << "\nRun " << is+1 << "/" << N_sample << "\tstate " << ic+1 << "/" << N_collect << " reached in " << 
                (clock() - start_step)/ (double) CLOCKS_PER_SEC << "s\tMBD=" << maxM(psi) << "\tP_decay=" << (double) (count_jump_decay)/(double) (N_step) << 
                "\tP_absorb=" << (double) (count_jump_absorb)/(double) (N_step) << endl << endl;
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