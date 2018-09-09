cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################

set(PROJECT_NAME CudaSoliditySHA3Solver)
# Output Variables
set(OUTPUT_DEBUG bin/Debug/)
set(OUTPUT_REL bin/Release/)

############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CXX)
find_package(CUDA 9.2 REQUIRED)

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

############## Artefacts Output #################
# Defines outputs , depending Debug or Release. #
#################################################

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
  set(CMAKE_EXECUTABLE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_REL}")
endif()

################# Flags ################
# Defines Flags for Windows and Linux. #
########################################

set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
  -std=c++11
  -Xptxas=-v,-warn-spills
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_32,code=sm_32
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_52,code=sm_52
  -gencode arch=compute_53,code=sm_53
  -gencode arch=compute_60,code=sm_60
  -gencode arch=compute_61,code=sm_61
  -gencode arch=compute_62,code=sm_62
  -gencode arch=compute_70,code=sm_70
)

if(NOT MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif(NOT MSVC)

################ Files ################
#   --   Add files to project.   --   #
#######################################

set(CPP_DIR_1 ./)
set(CPP_DIR_2 device)
set(CPP_DIR_3 uint256)
set(HEADER_DIR_1 )
set(HEADER_DIR_2 device)
set(HEADER_DIR_3 uint256)

file(GLOB SRC_FILES
  ${CPP_DIR_1}/*.cu
  ${CPP_DIR_1}/*.cpp
  ${CPP_DIR_2}/*.cpp
  ${CPP_DIR_3}/*.cpp
  ${HEADER_DIR_1}/*.h
  ${HEADER_DIR_2}/*.h
  ${HEADER_DIR_3}/*.h
)

cuda_add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
