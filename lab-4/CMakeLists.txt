cmake_minimum_required(VERSION 3.29)
project(lab-4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx -pthread -O3 -fopenmp")

add_executable(
        a.out
        mod_ops.cpp
        num_threads.cpp
        vector_mod.cpp
        test.cpp
        randomize.cpp
        performance.cpp
        entrypoint.cpp
)

# g++ -std=c++20 -mavx -pthread -O3 -fopenmp entrypoint.cpp mod_ops.cpp num_threads.cpp performance.cpp randomize.cpp test.cpp vector_mod.cpp