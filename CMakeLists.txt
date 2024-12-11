cmake_minimum_required(VERSION 3.10)

project(MyProject)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Specify Eigen3 include directory
set(EIGEN3_INCLUDE_DIR "/home/marta.rende/speciesTransport/eigen3")

# Add include directory
include_directories(${EIGEN3_INCLUDE_DIR})

# Add your source files
add_executable(run_speciesTransport 
    main.cpp 
    solve/solve.cpp 
    initialization/initialization.cpp 
    write/write.cpp
    solve/tools.cpp
)

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
