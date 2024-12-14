#ifndef SOLVE_H
#define SOLVE_H


struct SparseMatrix {
    int* row;     // Row indices (COO format)
    int* col;     // Column indices (COO format)
    double* value; // Non-zero values
    int nnz;       // Number of non-zero elements
};


void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt);
void computeBoundaries(double** Y, const int nx, const int ny);
//void fillMatrixA(SparseMatrix &A_sparse , const double dx, const double dy, const double D, const double dt, const int nx, const int ny);
extern void jacobiSolver(SparseMatrix &A_sparse, double *b, double *x, int n, int max_iter, double tol);
#endif // SOLVE_H