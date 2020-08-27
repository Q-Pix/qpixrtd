cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Qpix)

############################################################
# Create a library
############################################################

file(GLOB SOURCES "source/*.cpp")

##Generate the static library from the library sources
add_library(Qpix STATIC ${SOURCES})

## Add the Qpix includes
target_include_directories(Qpix
    PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
)



## find the ROOT package
find_package(ROOT REQUIRED)

## include ROOT header files
include(${ROOT_USE_FILE})

## link ROOT libraries
# link_libraries(${ROOT_LIBRARIES})

target_link_libraries(Qpix ${SOURCES})
target_link_libraries(Qpix ${ROOT_LIBRARIES})

