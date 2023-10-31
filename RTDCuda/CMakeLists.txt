## ---------------------------------------------------------
##  G4_QPIX | RTDCuda/CMakeLists.txt
##
##  CMake build script.
##   * Author: Kevin Keefe
##   * Creation date: 10 Oct 2023
## ---------------------------------------------------------



# target_include_directories(rtdcuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(rtdcuda ${CMAKE_PROJECT_NAME} ${Geant4_LIBRARIES})

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
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Define your source files
set(SOURCE_FILES RTDCudaMain.cpp)

# Add the CUDA source file
# list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
list(APPEND SOURCE_FILES RTDCuda.cu)

# Add the executable
add_executable(RTDCudaMain ${SOURCE_FILES})

target_link_libraries(RTDCudaMain ${ROOT_LIBRARIES})
target_link_libraries(RTDCudaMain ${QPixRTD_LIBRARIES})
target_link_libraries(RTDCudaMain QPixRTD)
target_include_directories(RTDCudaMain PUBLIC ${CMAKE_SOURCE_DIR}/source/)

set(CUDA_ARCHITECTURES 86)
set_target_properties(RTDCudaMain PROPERTIES
  CUDA_STANDARD 11
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)