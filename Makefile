all:
	g++  main.cpp solve/solve.cpp write/write.cpp initialization/initialization.cpp solve/tools.cpp -o run_speciesTransport

clean:
	rm run_speciesTransport