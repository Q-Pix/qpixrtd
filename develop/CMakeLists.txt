cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  EXAMPLE
  VERSION 0.1
  DESCRIPTION "develop"
  LANGUAGES CXX)

## find the ROOT package
find_package(ROOT REQUIRED)

## include ROOT header files
include(${ROOT_USE_FILE})

## find the Boost package
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

## add Q_PIX_RTD directories
include_directories($ENV{QPixRTDDir}/source)
link_directories($ENV{QPixRTDDir}/Library)

## executable
add_executable(develop develop.cpp)

## link libraries
target_link_libraries(develop ${Boost_LIBRARIES})
target_link_libraries(develop ${ROOT_LIBRARIES})
target_link_libraries(develop ${QPixRTD_LIBRARIES})
target_link_libraries(develop QPixRTD)
