#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include "write.h"

using namespace std;

// == cast data ==
string getString(double *data,  long size, int world_rank)
{
    string toWrite = "";
    for (int i = 0; i < size; i++)
    {
        toWrite += to_string(data[i]) + "\n";
    }
    return toWrite;
}
// == Write data to VTK file ==

using namespace std;
void writeDataVTK(const string filename, string *Y_part, string u_part, string v_part, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank, const int world_size, const int nSpecies)
{
    // == Create and open file with mpi ==
    MPI_File fh;
    string filename_all = "0000000" + to_string(step);
    reverse(filename_all.begin(), filename_all.end());
    filename_all.resize(7);
    reverse(filename_all.begin(), filename_all.end());
    filename_all = filename + filename_all + ".vtk";

    MPI_File_open(MPI_COMM_WORLD, filename_all.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Offset header_offset;
    // == rank 0 writes header ==
    if (world_rank == 0)
    {
        string header = "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";
        header += "DIMENSIONS " + to_string(nx) + " " + to_string(ny) + " 1\n";
        header += "X_COORDINATES " + to_string(nx) + " float\n";
        for (int j = 0; j < nx; j++)
        {
            header += to_string(j * dx) + "\n";
        }

        header += "Y_COORDINATES " + to_string(ny) + " float\n";
        for (int i = 0; i < ny; i++)
        {
            header += to_string(i * dy) + "\n";
        }

        header += "Z_COORDINATES 1 float\n0\nPOINT_DATA " + to_string(nx * ny) + "\n";
        header += "SCALARS Y float 1\nLOOKUP_TABLE default\n";
        MPI_File_write(fh, header.c_str(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        header_offset = header.size() * sizeof(char);
    }
    // share header offset with all process 
    MPI_Bcast(&header_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    MPI_Offset Y_offset = header_offset;
    MPI_Offset u_offset;
    MPI_Offset v_offset;

    // Write Y species data, one variable for each species
    for (int s = 0; s < nSpecies; s++)
    {
        MPI_Offset speciesHeaderSize;
        string speciesHeader = "\nSCALARS Y" + to_string(s) + " float 1\nLOOKUP_TABLE default\n";

        // Write header for the species if on rank 0
        if (world_rank == 0)
        {
            speciesHeaderSize = speciesHeader.size() * sizeof(char);
            MPI_File_write_at(fh, Y_offset, speciesHeader.c_str(), speciesHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
        }

        // Broadcast species header size
        MPI_Bcast(&speciesHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
        MPI_Offset speciesOffset = Y_offset + speciesHeaderSize;
        MPI_File_write_at(fh, speciesOffset, Y_part[s].c_str(), Y_part[s].size(), MPI_CHAR, MPI_STATUS_IGNORE);

        // Update offset after writing the current species data
        Y_offset = speciesOffset + Y_part[s].size();
        MPI_Bcast(&Y_offset, 1, MPI_OFFSET, world_size - 1, MPI_COMM_WORLD);
    }

    // Handle the u scalar data
    MPI_Offset uHeaderSize;
    if (world_rank == 0)
    {
        string uHeader = "\nSCALARS u float 1\nLOOKUP_TABLE default\n";
        uHeaderSize = uHeader.size() * sizeof(char);
        MPI_File_write_at(fh, Y_offset, uHeader.c_str(), uHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&uHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    u_offset = Y_offset + uHeaderSize;
    MPI_File_write_at(fh, u_offset, u_part.c_str(), u_part.size(), MPI_CHAR, MPI_STATUS_IGNORE);

    // Handle the v scalar data
    MPI_Offset vHeaderSize;
    if (world_rank == 0)
    {
        string vHeader = "\nSCALARS v float 1\nLOOKUP_TABLE default\n";
        vHeaderSize = vHeader.size() * sizeof(char);
        MPI_File_write_at(fh, u_offset + u_part.size(), vHeader.c_str(), vHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&vHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    v_offset = u_offset + uHeaderSize + u_part.size() + vHeaderSize;
    MPI_File_write_at(fh, v_offset, v_part.c_str(), v_part.size(), MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}
