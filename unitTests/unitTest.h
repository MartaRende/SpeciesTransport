#ifndef UNITTEST_H
#define UNITTEST_H

void runTestRowOffset(int nx, int ny, const char* testName) ;
void runTestfillMatrixA(int nx, int ny, double dx, double dy, double D, double dt, const char* testName);
void testJacobiSolver(int nx, int ny, int nnz,int * row, int *col , double * values, double * b, double* x, double *x_new);
#endif
