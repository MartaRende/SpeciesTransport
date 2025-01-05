main:
	nvcc -c main.cu initialization/initialization.cu solve/solve.cu solve/tools.cu solve/jacobi.cpp write/write.cpp  -I/usr/lib/x86_64-linux-gnu/openmpi/include
	mpic++ -o run_speciesTransport main.o initialization.o solve.o tools.o write.o jacobi.o -L/usr/local/cuda/lib64 -lcudart
test: 
	nvcc unitTests/computeRowOffsets.cu unitTests/main.cu unitTests/checkFillA.cu unitTests/checkJacobiSolver.cu  unitTests/checkFillB.cu solve/tools.cu -o run_speciesTransport
clean:
	rm run_speciesTransport