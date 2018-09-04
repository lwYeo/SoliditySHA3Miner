cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################

set(PROJECT_NAME OpenCLSoliditySHA3Solver)
# Output Variables
set(OUTPUT_DEBUG bin/Debug/)
set(OUTPUT_REL bin/Release/)

############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CXX)
find_package(OpenCL REQUIRED)

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

set(COMPILER_FLAGS " ")
set(LINKER_FLAGS " ")
set(ADDITIONAL_LIBRARIES "")

# gcc/g++ specific compile options
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(COMPILER_FLAGS "${COMPILER_FLAGS} -std=c++17")
    
  # Note: "rt" is not present on mingw
  if(UNIX)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(COMPILER_FLAGS " -g")
    endif()
    set(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} "rt")
  endif()
    
  set(COMPILER_FLAGS "${COMPILER_FLAGS} -m64")
  set(LINKER_FLAGS "${LINKER_FLAGS} -m64")
    
  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_GXX}")
  set(LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_GXX}")
  set(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_GXX})
elseif(MSVC)
  # Samples can specify additional libs/flags using EXTRA* defines
	add_definitions("/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /nologo")

  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_MSVC} ")
  set(LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_MSVC}  /SAFESEH:NO ")
  set(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_MSVC})
endif()

################ Files ################
#   --   Add files to project.   --   #
#######################################

set(CPP_DIR_1 ./)
set(CPP_DIR_2 device)
set(CPP_DIR_3 uint256)
set(HEADER_DIR_1 )
set(HEADER_DIR_2 uint256)
set(HEADER_DIR_3 device)
set(HEADER_DIR_4 device/adl_include)

file(GLOB SRC_FILES
  ${CPP_DIR_1}/*.cpp
  ${CPP_DIR_2}/*.cpp
  ${CPP_DIR_3}/*.cpp
  ${HEADER_DIR_1}/*.h
  ${HEADER_DIR_2}/*.h
  ${HEADER_DIR_3}/*.h
  ${HEADER_DIR_4}/*.h
)

include_directories($ENV{AMDAPPSDKROOT}/include)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

if(MSVC)
  target_link_libraries(${PROJECT_NAME} OpenCL.lib ${ADDITIONAL_LIBRARIES})
else()
  target_link_libraries(${PROJECT_NAME} $ENV{AMDAPPSDKROOT}/lib/x86_64/libOpenCL.so ${ADDITIONAL_LIBRARIES})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" COMPILE_FLAGS ${COMPILER_FLAGS} LINK_FLAGS ${LINKER_FLAGS})

# Copy *.cl files to output directory
file(GLOB _CL_FILE *.cl)
set(CL_FILES ${_CL_FILE})

foreach(cl_file ${CL_FILES})
  add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${cl_file} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  )
endforeach(cl_file)
