all:
	nvcc -c main.cu initialization/initialization.cu solve/solve.cu solve/tools.cu solve/jacobi.cpp write/write.cpp -I/usr/lib/x86_64-linux-gnu/openmpi/include
	mpic++ -o run_speciesTransport main.o initialization.o solve.o tools.o write.o jacobi.o -L/usr/local/cuda/lib64 -lcudart
clean:
	rm run_speciesTransport