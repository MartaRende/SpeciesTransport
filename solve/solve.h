#ifndef SOLVE_H
#define SOLVE_H



void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt);
void computeBoundaries(double** Y, const int nx, const int ny);
double** invertMatrix(double** matrix, int n);
void fillMatrixA(double **A, const double dx, const double dy, const double D, const double dt, const int nx, const int ny);
#endif // SOLVE_H