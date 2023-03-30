## ---------------------------------------------------------
##  G4_QPIX | app/CMakeLists.txt
##
##  CMake build script.
##   * Author: Kevin Keefe
##   * Creation date: 30 Mar 2023
## ---------------------------------------------------------


add_executable(rtdfriend RTDFriend.cpp)
target_include_directories(rtdfriend PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(rtdfriend ${CMAKE_PROJECT_NAME} ${Geant4_LIBRARIES})

## Add ROOT
target_link_libraries(rtdfriend ${ROOT_LIBRARIES})