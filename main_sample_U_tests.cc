//
//  main.cpp
//  annealing 2D Bose Hubbard model using MPI
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


using namespace std;
using namespace itensor;
using namespace Eigen;

void save_data(){
    
}

int main ( int argc, char *argv[] )
{
        //---------------------------------------------------------
    //----------------------initialize MPI---------------------
    Environment env(argc,argv);
    
    //---------------------------------------------------------
    //---------parse parameters from input file ---------------
    auto input = InputGroup(argv[1],"input");
    
    auto N_spin = atoi(argv[2]);
    auto N_sample = atoi(argv[3]);
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",200);

    // probability of direct measurement
    auto p = input.getReal("p",0.);

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //-----------------------------------------------------------
    // START
    
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    auto sites = SpinHalf(N_spin, args);
    
    
    for (int is = 0; is<N_sample; is++){
        
        // sample Haar U and add decays
        MatrixXcf U = random_U_haar(N_spin);
        
        // to test code
        //MatrixXcf U = MatrixXcf::Identity(N_spin,N_spin);
        
        vector<MPO> C;
        add_decay_U(C, sites, U, sqrt(1.-p));

        // initial state
        auto state = InitState(sites);
        for(int i = 1; i <= N_spin; ++i)
        {
            state.set(i,"Up");
        }
        auto psi = MPS(state);

        for (int i_jump = 0; i_jump<N_spin; i_jump++){
            vector<double> Pc_i;
            vector<MPS> save_psi;
            
            double sum_Pc = 0.;
            
            for (int ic = 0; ic<N_spin; ic++){
                MPS c_psi = applyMPO(C[ic],psi,args);
                double Pc = real(overlapC(c_psi, c_psi));
                
                sum_Pc += Pc;

                //store
                save_psi.push_back(c_psi);
                Pc_i.push_back(Pc);
                cout << "P: " << i_jump << " " << ic << " " << Pc << endl;
            }
            
            // select jump and apply to psi
            double e = ((double) rand() / (RAND_MAX))*sum_Pc;
            double sum_pj = 0.;
            for (int ic=0; ic<N_spin; ic++){
                sum_pj += Pc_i[ic];
                if (sum_pj>e){
                    psi = save_psi[ic];
                    psi.position(1,args);
                    normalize(psi);
                    cout << "pick: " << ic << ", sum: " << sum_Pc << endl;
                    for (int j=1; j<=N_spin; j++){
                        psi.position(j);
                
                        ITensor ket = psi.A(j);
                        ITensor bra = dag(prime(ket,Site));
                
                        ITensor Szjop = sites.op("projUp",j);

                        //take an inner product 
                        auto szj = (bra*Szjop*ket).real();
                        cout << szj << " ";
                    }
                    cout << endl << endl;
                    break;
                }
            }
            
            
        }
        
    }
        
    
    
    return 0;
    
}