#include <complex>
#include <vector>
#include <cmath>
#include "itensor/all.h"
#include "itensor/util/parallel.h"

using namespace std;
using namespace itensor;

#ifndef func
#define func

//print occupations
void print_occupations(MPS& psi,Args args){
    auto sites = psi.sites();
    
    double total = 0.;
    for (int i=1; i<=psi.N(); i++){
        ITensor projUp_i = sites.op("projUp",i);
        
        // normalize and get local ket and bra
        psi.position(i,args); normalize(psi);
        ITensor ket = psi.A(i); ITensor bra = dag(prime(ket,Site));
        double occ = (bra*projUp_i*ket).real();
        
        cout <<  occ << " ";
        total += occ;
    }
    cout << " total: " << total << endl;
}

void save_entropies(const vector<double>& Sv, string files, int i_jump){
    
    auto file_save = files + "_" + to_string(i_jump) + ".txt";
    ofstream file;
    file.open(file_save, ofstream::out | ofstream::app);
    
    for (unsigned int i=0; i<Sv.size(); i++){
        file << Sv[i] << " ";
    }
    
    file << endl;

}

void SelectAndApplyJump(MPS& psi, const vector<MPO>& C, Args args){
    
    int N_channel = C.size();
    
    // save probabilities and states
    vector<double> Pc_i;
    vector<MPS> save_psi;
    
    // total sum of probabilities
    double sum_Pc = 0.;
    
    // make sure it's normalized
    psi.position(1,args);
    normalize(psi);
    
    // loop over jumps, save probabilities and states
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
            break;
        }
    }
}

void NLayerApplyJump(MPS& psi, string op, int N_layer, Args args){
    
    vector<MPO> C;
    
     // random offset 0 or 1, where to start first 2x2 block
    int offset = rand() % 2;
    MatrixXcf U = random_U_haar_layer(psi.N(), N_layer, offset);
    add_jump_U(C, psi.sites(), U, op);

    // select jump and apply to psi
    SelectAndApplyJump(psi, C, args);
    // normalize
    psi.position(1,args);
    normalize(psi);
}

void SelectLayerAndApplyJump(MPS& psi,  string op, double p, Args args){
    
    // select which layer and add jumps to vector C
    double coeff = p;
    double pl = ((double) rand() / (RAND_MAX));
    
    double sum_coeff = p;
    int N_layer = 0;
    while (sum_coeff < pl){
        coeff *= (1.-p);
        sum_coeff += coeff;
        N_layer++;
    }
    
    // apply N_layer jump
    NLayerApplyJump(psi, op, N_layer, args);
}

// perform spin measurements with probability p
void SelectAndApplyMeasurements(MPS& psi, double p, Args args){
    
    auto sites = psi.sites();

    // loop through chain, measure every spin with probability p
    for (int i=1; i<=psi.N(); i++){
        bool click = p > (double) rand() / (RAND_MAX);
        if (click){
            // to measure
            ITensor projUp_i = sites.op("projUp",i);
            ITensor projDn_i = sites.op("projDn",i);
            
            // normalize and get local ket and bra
            psi.position(i,args); normalize(psi);
            ITensor ket = psi.A(i); ITensor bra = dag(prime(ket,Site));
            
            // excited level occupation
            auto N_i = (bra*projUp_i*ket).real();
            
            // apply projection on up or down state
            double s = (double) rand() / (RAND_MAX);
            ITensor newA;
            if (s<N_i){
                newA = projUp_i*ket;
            }
            else{
                newA = projDn_i*ket;
            }
            newA.noprime();
            psi.setA(i,newA);
        }
    }
    
    // normalize again
    psi.position(1,args);
}
        
            
            
#endif