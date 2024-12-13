#ifndef SOLVE_H
#define SOLVE_H

#include <vector>

struct SparseMatrix {
    std::vector<int> row;     // Row indices (COO format)
    std::vector<int> col;     // Column indices (COO format)
    std::vector<double> value; // Non-zero values
};


void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt);
void computeBoundaries(double** Y, const int nx, const int ny);
void fillMatrixA(SparseMatrix &A_sparse , const double dx, const double dy, const double D, const double dt, const int nx, const int ny);
void jacobiSolver(SparseMatrix &A_sparse, double* b, double* x, int size, int maxIterations, double tolerance);
#endif // SOLVE_H