############################################################
# Create the QPixRTD library
############################################################

file(GLOB SOURCES "*.cpp")

# # generate ROOT dictionary
ROOT_GENERATE_DICTIONARY(QPixRTDDict ROOTFileManager.h LINKDEF LinkDef.h)

##Generate the static library from the library sources
add_library(QPixRTD SHARED ${SOURCES} QPixRTDDict.cxx)

target_link_libraries(QPixRTD ${ROOT_LIBRARIES})
target_link_libraries(QPixRTD ${Boost_LIBRARIES})

