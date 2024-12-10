#include "solve.h"
#include <stdio.h>

using namespace std;

void computeBoundaries(double** Y, const int nx, const int ny){
     for (int i = 0; i < nx ; i++){
        Y[i][ny-1] = 0.0;
        Y[i][0] = 0.0;
    }

    for (int j = 0; j < ny; j++){
        Y[0][j] = 0.0;
        Y[nx-1][j] = 0.0;
    }

}

void solveSpeciesEquation(double** Y,double** u,double** v, const double dx,const double dy,double D,const int nx,const int ny, const double dt){
double** Y_n = new double*[nx]; // Previous Y
for (int i = 0; i < nx; i++) {
    Y_n[i] = new double[ny];
    for (int j = 0; j < ny; j++){
        Y_n[i][j] = Y[i][j];
    } 
}
    double dt1 = 0.01; // Temporal step [s]
    double rho = 1.204; // [kg/m^3]
    double cp = 1004.0; // [J/K/kg] -> [m^2/K/s^2]
    double lambda = 0.026; // [W/m/K] -> [kg.m/s^3/K]
for (int i = 1; i < nx-1; i++){
    for (int j = 1; j < ny-1; j++){
            
            Y[i][j] = Y_n[i][j] + dt*D  *( (Y_n[i+1][j] - 2.0 *Y_n[i][j] + Y_n[i-1][j])/dx/dx + (Y_n[i][j+1] - 2.0 *Y_n[i][j] + Y_n[i][j-1])/dy/dy );


            if (u[i][j] < 0.0) { 
                Y[i][j] -= dt * ( u[i][j]*(Y_n[i+1][j] - Y_n[i][j])/dx);
            }
            else {
                Y[i][j] -= dt * ( u[i][j]*(Y_n[i][j] - Y_n[i-1][j])/dx);
            }

            if (v[i][j] < 0.0) { 
                Y[i][j] -= dt * ( v[i][j]*(Y_n[i][j+1] - Y_n[i][j])/dy);
            }
            else {
                Y[i][j] -= dt * ( v[i][j]*(Y_n[i][j] - Y_n[i][j-1])/dy);
            }
    
    }

}
computeBoundaries(Y,nx,ny); 
    for (int i = 0; i < nx; ++i) {
        delete[] Y_n[i];
    }
    delete[] Y_n;


}