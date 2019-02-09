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

project(${PROJECT_NAME})
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
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_52,code=sm_52
  -gencode arch=compute_61,code=sm_61
)

if(NOT MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif(NOT MSVC)

################ Files ################
#   --   Add files to project.   --   #
#######################################

set(DIR_1 ./)
set(DIR_2 ./device)

file(GLOB SRC_FILES
  ${DIR_1}/*.cu
  ${DIR_1}/*.cpp
  ${DIR_2}/*.cpp
  ${DIR_1}/*.h
  ${DIR_2}/*.h
)

cuda_add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
