cmake_minimum_required(VERSION 3.5)

project(MnistInCProject)


set(CMAKE_C_STANDARD 11)


set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GENANN_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/genann_header)
set(GENANN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/genann_src)


file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.c ${GENANN_SRC_DIR}/*.c)


include_directories(${GENANN_HEADER_DIR})


add_executable(mnist_in_c ${SRC_FILES})
