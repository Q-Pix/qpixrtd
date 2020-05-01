cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

project(Qpix)

############################################################
# Create a library
############################################################

file(GLOB SOURCES "source/*.cc")

#Generate the static library from the library sources
add_library(Qpix STATIC ${SOURCES})


target_include_directories(Qpix
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)
