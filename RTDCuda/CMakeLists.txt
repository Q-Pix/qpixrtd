## ---------------------------------------------------------
##  RTDCuda | RTDCuda/CMakeLists.txt
##
##  CMake build script.
##   * Author: Kevin Keefe
##   * Creation date: 10 Oct 2023
## ---------------------------------------------------------



# target_include_directories(rtdcuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(rtdcuda ${CMAKE_PROJECT_NAME} ${Geant4_LIBRARIES})

project(RTDCuda)
# add_library(rtdcudaLib RTDCuda.cu)

# enable_language(CUDA)
# include(CheckLanguage)
# check_language(CUDA)

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 11)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()

# set_target_properties(rtdcudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

## Link ROOT
# target_link_libraries(rtdcudaLib ${ROOT_LIBRARIES})

## gpt
enable_language(CUDA)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

# Define your source files
set(SOURCE_FILES RTDCudaMain.cpp RTDCudaFileManager.cpp)

# Add the CUDA source file
# list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND SOURCE_FILES RTDCuda.cu RTDThrust.cu)

# Add the executable
add_executable(RTDCudaMain ${SOURCE_FILES})
target_include_directories(RTDCudaMain PUBLIC ${CMAKE_SOURCE_DIR}/RTDCuda/)

target_link_libraries(RTDCudaMain ${ROOT_LIBRARIES})
target_link_libraries(RTDCudaMain QPixRTD)
target_link_libraries(RTDCudaMain Thrust)

set(CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "-Wall")

set_target_properties(RTDCudaMain PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
  CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}
  CMAKE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
  CUDA_NVCC_FLAGS "-Wall"
)