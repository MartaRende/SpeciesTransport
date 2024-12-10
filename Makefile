all:
	g++ main.cpp solve/solve.cpp write/write.cpp initialization/initialization.cpp -o run_speciesTransport
clean:
	rm run_speciesTransport