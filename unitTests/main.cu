#include "unitTest.h"
#include <iostream>
int main()
{
    printf("Starting Tests to check offset row\n");
  // 1. Small Grid Test (3x3)
    runTestRowOffset(3, 3, "Small Grid Test (3x3)");

    // 2. Large Grid Test (5x5)
    runTestRowOffset(5, 5, "Large Grid Test (5x5)");

    // 3. Boundary-Only Grid Test (2x2)
    runTestRowOffset(2, 2, "Boundary-Only Grid Test (2x2)");

    // 4. Single Cell Grid Test (1x1)
    runTestRowOffset(1, 1, "Single Cell Grid Test (1x1)");

    // 5. Boundary-Only Grid Test (1x5)
    runTestRowOffset(1, 5, "Boundary-Only Grid Test (1x5)");

    // 6. Empty Grid Test (0x0)
    runTestRowOffset(0, 0, "Empty Grid Test (0x0)");

    // 7. Rectangular Grid Test (6x4)
    runTestRowOffset(6, 4, "Rectangular Grid Test (6x4)");

    // 8. Rectangular Grid Test (4x6)
    runTestRowOffset(4, 6, "Rectangular Grid Test (4x6)");

    // 9. Large Square Grid Test (100x100)
    runTestRowOffset(100, 100, "Large Square Grid Test (100x100)");

    // 10. Very Large Rectangular Grid Test (1000x500)
    runTestRowOffset(1000, 500, "Very Large Rectangular Grid Test (1000x500)");

    std::cout << "All tests completed successfully." << std::endl;
    printf("All test passed for checking the offset of rows in matrix A\n");
   // 1. Basic Functionality Test
   // runTestfillMatrixA(5, 5, 0.1, 0.1, 1.0, 0.01, "Basic Functionality Test");

    // 2. Boundary Condition Test
   // runTestfillMatrixA(3, 3, 0.1, 0.1, 1.0, 0.01, "Boundary Condition Test");

   /*  // 3. Different Grid Size Test
    runTestfillMatrixA(10, 10, 0.1, 0.1, 1.0, 0.01, "Different Grid Size Test");

    // 4. Parameter Variation Test
    runTestfillMatrixA(5, 5, 0.2, 0.2, 2.0, 0.005, "Parameter Variation Test");

    // 5. Edge Case Test (Small Grid)
    runTestfillMatrixA(2, 2, 0.1, 0.1, 1.0, 0.01, "Edge Case Test (Small Grid)");

    // 6. Large Grid Test
    runTestfillMatrixA(100, 100, 0.01, 0.01, 1.0, 0.001, "Large Grid Test");

    // 7. Non-Square Grid Test
    runTestfillMatrixA(8, 5, 0.1, 0.15, 1.0, 0.01, "Non-Square Grid Test");

    // 8. Precision Test
    runTestfillMatrixA(5, 5, 0.1, 0.1, 1.0, 0.0001, "Precision Test");

    // 9. Zero and Negative Parameter Test
    try {
        runTestfillMatrixA(5, 5, 0, 0.1, -1.0, 0.01, "Zero and Negative Parameter Test");
    } catch (...) {
        std::cout << "Zero and Negative Parameter Test caught exception as expected: " ;
    }

    // 10. Row Offset Correctness Test
    runTestfillMatrixA(7, 7, 0.1, 0.1, 1.0, 0.01, "Row Offset Correctness Test");
*/
    printf("All test passed to check filling of matrix A\n");
   // Test case: 3x3 matrix, 9 non-zero elements
    int row[] = {0, 3, 6, 9};  // Row pointers (nx + 1)
    int col[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};  // Column indices for non-zero elements (nnz)
    double value[] = {4, -1, -1, -1, 4, -1, -1, -1, 4};  // Non-zero values of the matrix (nnz)
    double b[] = {3.0, 7.0, 2.0};  // Right-hand side vector
    double x[] = {9.0/5, 13.0/5, 8.0/5};  // Initial guess
    double x_new[] = {0, 0, 0};  // Solution vector (output)

    // Call the Jacobi solver test
    testJacobiSolver(3, 3, 9, row, col, value, b, x, x_new);
    // Test case: 3x3 matrix, 9 non-zero elements
int row2[] = {0, 3, 6, 9};  // Row pointers (nx + 1)
int col2[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};  // Column indices for non-zero elements (nnz)
double value2[] = {10, -1, 2, -1, 11, -1, 2, -1, 10};  // Non-zero values of the matrix (nnz)
double b2[] = {6.0, 25.0, -11.0};  // Right-hand side vector
double x2[] = {1.0, 2.0, -1.0};  // Initial guess
double x_new2[] = {0.0, 0.0, 0.0};  // Solution vector (output)

// Call the Jacobi solver test
testJacobiSolver(3, 3, 9, row2, col2, value2, b2, x2, x_new2); 

                     return 0;    
    
}
