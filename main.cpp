#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "solve/solve.h"
#include "write/write.h"
#include "initialization/init.h"



int main(){

    //default paramenters 
    double D = 0.010; // possible values from 0.001 to 0.025

    int nx = 800; int ny = 800; // Number of cells in each direction 
    double Lx = 1.0; double Ly = 1.0; // Square domain [m]
    double dx = 0.0075; double dy = 0.0075; // Spatial step [m]
    
    // == Temporal ==
    double tFinal = 2.0; 
    double dt = 0.0005; 
    int nSteps = int(tFinal/dt); 
    double time = 0.0;

    //array initialization
    double** Y = new double*[nx];
    double** u = new double*[nx];
    double** v = new double*[nx];
    for(int i = 0 ; i < nx; i++){
        Y[i] = new double[ny]; 
        u[i] = new double[ny]; 
        v[i] = new double[ny];
    }
    
    Initialization(Y,u,v, nx, ny, dx, dy); // Initialize the temperature field inside the domain
    computeBoundaries(Y, nx, ny); 

    // == Output ==
    string outputName =  "output/speciesTransport_";
    int count = 0;

    // == First output == 
    // Write data in VTK format
    writeDataVTK(outputName, Y,u,v, nx, ny, dx, dy, count++);

    // Loop over time
    for (int step = 1; step <= nSteps; step++){

        // Solve the thermal equation (energy) to compute the temperature at the next time
        solveSpeciesEquation(Y,u,v,dx,dy,D,nx,ny,dt);

        // Write output every 100 iterations
        if (step%100 == 0){
            writeDataVTK(outputName, Y,u,v, nx, ny, dx, dy, count++);
        }

    }
    for (int i = 0; i < nx; i++) {
    delete[] u[i];
    delete[] v[i];
}
delete[] u;
delete[] v;
  return 0; 


}