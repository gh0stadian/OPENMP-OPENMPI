cmake_minimum_required(VERSION 3.17)
project(paralpr)

set(CMAKE_CXX_STANDARD 17)

add_executable(paralpr main.cpp)

find_package(OpenMP)
find_package(PNG)
find_package(MPI)
if(OpenMP_CXX_FOUND AND PNG_FOUND AND MPI_CXX_FOUND)
    target_link_libraries(paralpr PUBLIC OpenMP::OpenMP_CXX PNG::PNG MPI::MPI_CXX)
endif()
