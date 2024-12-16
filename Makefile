all:
	nvcc  main.cu solve/solve.cu write/write.cpp initialization/initialization.cu solve/tools.cu -o run_speciesTransport
	
clean:
	rm run_speciesTransport