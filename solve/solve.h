#ifndef SOLVE_H
#define SOLVE_H



void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt);
void computeBoundaries(double** Y, const int nx, const int ny);
//void fillMatrixA(double *A, const double dx, const double dy, const double D, const double dt, const int nx, const int ny);
void jacobiSolver(double* A, double* b, double* x, int size, int maxIterations, double tolerance);
#endif // SOLVE_H