cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(QPixRTD)

## find the Boost package
set(Boost_NO_BOOST_CMAKE ON)
FIND_PACKAGE(Boost 1.61 COMPONENTS system filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

## find the ROOT package
find_package(ROOT REQUIRED)

## include ROOT header files
include(${ROOT_USE_FILE})

## link ROOT libraries
# link_libraries(${ROOT_LIBRARIES})

include_directories(${CMAKE_SOURCE_DIR}/source)

add_subdirectory(source)
add_subdirectory(EXAMPLE)
add_subdirectory(RTDFriend)
add_subdirectory(CurrentGen)