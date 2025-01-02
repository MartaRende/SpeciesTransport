#ifndef SOLVE_H
#define SOLVE_H


struct SparseMatrix {
     int* row;     // Row indices (COO format)
     int* col;     // Column indices (COO format)
    double* value; // Non-zero values
     int nnz;       // Number of non-zero elements
};


void solveSpeciesEquation(double *Y, 
                          const double dx, const double dy, double D,
                          const  int nx, const  int ny, const double dt,double * d_u, double * d_v, double * d_Yn, double * d_x, double * d_x_new, double * d_b_flatten, double * d_values,  int * d_column_indices,  int * d_row_offsets, int world_rank);
//void fillMatrixA(SparseMatrix &A_sparse , const double dx, const double dy, const double D, const double dt, const int nx, const int ny);
void jacobiSolverSparseMPI(SparseMatrix &A, double *b, double *x,  int n,  int max_iter, double tol);

#endif // SOLVE_H