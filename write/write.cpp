#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "write.h"

using namespace std;

// Write data to VTK file  
void writeDataVTK(const string filename, double*** Y, double** u, double** v, const int nx, const int ny, const double dx, const double dy, const int step, const int nSpecies){

    // Create the filename 
    string filename_all = "0000000"+to_string(step);
    reverse(filename_all.begin(), filename_all.end());
    filename_all.resize(7);
    reverse(filename_all.begin(), filename_all.end());
    filename_all = filename + filename_all + ".vtk";

    cout << "Writing data into " << filename_all << "\n";

    // Setting open file 
    ofstream myfile;
    myfile.open(filename_all);
    
    // Write header of vtk file 
    myfile << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";

    myfile << "DIMENSIONS " << nx << " " << ny << " 1\n";
    myfile << "X_COORDINATES " << nx << " float\n";
    for (int i = 0; i < nx; i++){myfile << i*dx << "\n";}

    myfile << "Y_COORDINATES " << ny << " float\n";
    for (int j = 0; j < ny; j++){myfile << j*dy << "\n";}

    myfile << "Z_COORDINATES 1 float\n";
    myfile << "0\n";
    myfile << "POINT_DATA " << nx*ny << "\n";

    // Write the Y values, one variable for each species
    for( int s = 0; s < nSpecies ; s++){
    myfile << "SCALARS Y"<< s <<" float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            myfile << Y[s][i][j] << "\n";
        }
    }
}
    // Write the x velocity values 
    myfile << "\nSCALARS u float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            myfile << u[i][j] << "\n";
        }
    }

    // Write the y velocity values
    myfile << "\nSCALARS v float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            myfile << v[i][j] << "\n";
        }
    }


    // Close file
    myfile.close();

}
