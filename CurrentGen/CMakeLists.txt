cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  CurrentGen
  VERSION 0.1
  DESCRIPTION "CurrentGen"
  LANGUAGES CXX)

## find the ROOT package
find_package(ROOT REQUIRED)

## include ROOT header files
include(${ROOT_USE_FILE})

## find the Boost package
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.61 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

## add Q_PIX_RTD directories
include_directories($ENV{QPixRTDDir}/source)
link_directories($ENV{QPixRTDDir}/Library)

## executable
add_executable(CurrentGen CurrentGen.cpp)

## link libraries
target_link_libraries(CurrentGen ${Boost_LIBRARIES})
target_link_libraries(CurrentGen ${ROOT_LIBRARIES})
target_link_libraries(CurrentGen ${QPixRTD_LIBRARIES})
target_link_libraries(CurrentGen QPixRTD)
