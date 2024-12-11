#ifndef SOLVE_H
#define SOLVE_H
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;


void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt);
void computeBoundaries(double** Y, const int nx, const int ny);
double** invertMatrix(double** matrix, int n);
double determinant(double** A, int n);
MatrixXd computePseudoinverse(MatrixXd& A);
#endif // SOLVE_H