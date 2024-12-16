#ifndef WRITE_H
#define WRITE_H

#include <string>
#include <vector>

using namespace std;

void writeDataVTK(const string filename, string *Y_part, string u_part, string v_part, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank, const int world_size, const int nSpecies);
string getString(double *data, long size, int world_rank);
#endif // WRITE_H
