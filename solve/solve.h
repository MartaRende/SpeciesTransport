#ifndef SOLVE_H
#define SOLVE_H




void solveSpeciesEquation(
                          const double dx, const double dy, double D,
                          const  int nx, const  int ny, const double dt,double * d_u, double * d_v, double * d_Yn, double * d_x, double * d_x_new, double * d_b, double * d_values,  int * d_column_indices,  int * d_row_offsets, int world_rank);

#endif // SOLVE_H