cmake_minimum_required(VERSION 3.29)
project(lab-3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx -pthread -O3 -fopenmp")

add_executable(lab-3 main.cpp)