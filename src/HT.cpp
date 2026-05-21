#include <iostream>
#include "HT.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <bit>

using namespace std;
using namespace Eigen;

void multiply_HT(double* __restrict y, int n)
{
    if(n<=2048)
    {
    for(int h=1; h<n; h=h*2)
    {
        for(int i=0; i<n; i+=2*h)
        {
            #pragma omp simd
            for(int j=i; j<i+h; j++)
            {
                double s=y[j];
                double t=y[j+h];
                y[j]=s+t;
                y[j+h]=s-t;
            }
        }
    }
    return;
    }

    int h=n/2;
    multiply_HT(y,h);
    multiply_HT(y+h,h);

    #pragma omp simd
    for(int i=0; i<h; i++)
    {
        double s=y[i];
        double t=y[i+h];
        y[i]=s+t;
        y[i+h]=s-t;
    }

}

std::pair<MatrixXd, VectorXd> HT(const MatrixXd& A,const VectorXd& b)
{
    //I do not multiply by the constant factor sqrt(n/r) since it gets cancelled in the end.
    int n=A.rows();
    int d=A.cols();
    unsigned int n_padded=bit_ceil((unsigned)n);

    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<> diag(0, 1);
    VectorXd diagonal(n);

    for(int i=0; i<n; ++i)
    {
        diagonal(i)=diag(mt);
    }

    diagonal.array()=2*diagonal.array()-1;

    VectorXd y_f=VectorXd::Zero(n_padded);
    y_f.head(n)=b;
    y_f.head(n).array() *= diagonal.array();
    multiply_HT(y_f.data(), n_padded);

    MatrixXd X_f(n_padded,d);
    #pragma omp parallel
    {
    
    #pragma omp for
    for(int j=0; j<d; j++)
    {
        X_f.col(j).tail(n_padded-n).setZero();
        const double* col_ptr = &A(0, j);
        double* temp_ptr = X_f.col(j).data();
        const double* diag_ptr = diagonal.data();

        for(int i=0; i<n; i++)
        {
            temp_ptr[i] = col_ptr[i] * diag_ptr[i];
        }

        multiply_HT(temp_ptr, n_padded);
    }
    }
    return{X_f, y_f};
}
