// Libraries
#include <iostream>
#include <string>

// == User lib ==
#include "initialization/init.h"
#include "solve/solve.h"
#include "write/write.h"

// Namespace
using namespace std;

// Thermal Solver 
int main()
{

    // Data Initialization

    // == Spatial == 
    int nx = 1000; int ny = 1000; // Number of cells in each direction 
    double Lx = 1.0; double Ly = 1.0; // Square domain [m]
    double dx = Lx / (nx-1); double dy = Ly / (ny-1); // Spatial step [m]
    
    // == Temporal ==
    double tFinal = 100.0; // Final time [s]
    double dt = 0.01; // Temporal step [s]
    int nSteps = int(tFinal/dt); // Number of steps to perform
    double time = 0.0; // Actual Simulation time [s]

    // == Physics (Air) == 
    double rho = 1.204; // [kg/m^3]
    double cp = 1004.0; // [J/K/kg] -> [m^2/K/s^2]
    double lambda = 0.026; // [W/m/K] -> [kg.m/s^3/K]
    double** T = new double*[nx]; // Temperature
    bool** isComputed = new bool*[nx]; // Local indicator if we have to compute or not the energy
    for (int i = 0; i < nx; ++i) {
        T[i] = new double[ny];
        isComputed[i] = new bool[ny];
    }
    Initialization(T, isComputed, nx, ny, dx, dy); // Initialize the temperature field inside the domain
    computeBoundaries(T, nx, ny); 

    // == Output ==
    string outputName =  "output/energy_";
    int count = 0;

    // == First output == 
    // Write data in VTK format
    writeDataVTK(outputName, T, nx, ny, dx, dy, count++);

    // Loop over time
    for (int step = 1; step <= nSteps; step++){

        time += dt; // simulation time increases
        cout << "\nStarting iteration step " << step << "/"<< nSteps << "\tTime " << time << "s\n"; 

        // Solve the thermal equation (energy) to compute the temperature at the next time
        solveThermalEquationExplicit(T, isComputed, nx, ny, dx, dy, dt, lambda, rho, cp);

        // Write output every 100 iterations
        if (step%100 == 0){
            writeDataVTK(outputName, T, nx, ny, dx, dy, count++);
        }

    }


    // Deallocate memory
    for (int i = 0; i < nx; ++i) {
        delete[] T[i], isComputed[i];
    }
    delete[] T, isComputed;

    return 0;
}
