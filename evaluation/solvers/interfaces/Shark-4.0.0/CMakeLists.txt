cmake_minimum_required(VERSION 3.24)
project(Shark_4_0_0)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost COMPONENTS serialization)
include_directories(/home/fe/yolcu/Documents/Code/THESIS/evaluation/solvers/libs/Shark-4.0.0/include)
include_directories(/home/fe/yolcu/Documents/Code/THESIS/evaluation/solvers/libs/Shark-4.0.0/build/include)
link_directories(/home/fe/yolcu/Documents/Code/THESIS/evaluation/solvers/libs/Shark-4.0.0/build/lib/)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Shark_4_0_0 main.cpp)

target_link_libraries(Shark_4_0_0 ${Boost_LIBRARIES})
target_link_libraries(Shark_4_0_0 /usr/lib/x86_64-linux-gnu/libcblas.so)
target_link_libraries(Shark_4_0_0 libshark.a)


