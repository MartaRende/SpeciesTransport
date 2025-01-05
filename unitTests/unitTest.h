#ifndef UNITTEST_H
#define UNITTEST_H

void runTestRowOffset(int * row, int nx, int ny, const char* testName) ;
void runTestfillMatrixA(int* row_offset,double* exp_values,int nx, int ny, double dx, double dy, double D, double dt, const char *testName);
void testJacobiSolver(int nx, int ny, int nnz,int * row, int *col , double * values, double * b, double* x, double *x_new, const char *testName );
void testFillb(int nx, int ny, double dt, double dx, double dy, double *u, double *v, double *Yn, double * b_expeted, double *b, const char *testName);
#endif
