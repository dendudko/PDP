cmake_minimum_required(VERSION 3.30)
project(lab-5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx -pthread -O3")

add_executable(lab-5 main.cpp)
# g++ -std=c++2a -mavx -pthread -O3 main.cpp