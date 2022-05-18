#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>
#include "itensor/all.h"
#include "itensor/util/parallel.h"

using namespace std;
using namespace itensor;
using namespace Eigen;

#ifndef decay
#define decay

AutoMPO H_drive(SiteSet sites, double Omega, double delta=0., double gamma_decay=1., double gamma_absorb=0.){
    auto H = AutoMPO(sites);
    
    // single site drive and decay
    for (int b=1; b<=sites.N(); b++){
        H += -delta, "Sz",b;
        H += Omega, "Sx",b;
        
        // anti Herm part decay
        H += -Cplx_i*gamma_decay/2.0, "projUp",b;
        // anti Herm part absorb
        H += -Cplx_i*gamma_absorb/2.0, "projDn",b;
    }
    
    return H;
}

void add_decay_U(vector<MPO>& Cv, SpinHalf sites, MatrixXcf U, double gamma = 1.){
    
    if (gamma > 1E-4){
        int N = sites.N();
    
        for (int i=0; i<N; i++){
            
            AutoMPO C(sites);
    
            for (int j=0; j<N; j++){
                complex<double> coeff = U(i,j);
                
                if (abs(coeff) > 1e-6){
                    C += gamma*(real(coeff) + Cplx_i*imag(coeff)), "S-", j+1;
                }
            }
            auto C_MPO = MPO(C);
            Cv.push_back(C_MPO);
            
        }
    }
}

void add_jump_U(vector<MPO>& Cv, SiteSet sites, MatrixXcf U, string op = "S-", double gamma = 1.){
    
    if (gamma > 1E-4){
        int N = sites.N();
    
        for (int i=0; i<N; i++){
            
            AutoMPO C(sites);
    
            for (int j=0; j<N; j++){
                complex<double> coeff = U(i,j);
                
                if (abs(coeff) > 1e-6){
                    C += gamma*(real(coeff) + Cplx_i*imag(coeff)), op, j+1;
                }
            }
            auto C_MPO = MPO(C);
            Cv.push_back(C_MPO);
            
        }
    }
}




#endif
