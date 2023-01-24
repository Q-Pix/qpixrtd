#!/bin/sh

###########################################
# build.sh
#
# Author: Dave Elofson
# Created: 2022-03-31
##########################################

# This script will make the build directory, then run cmake and make. Should only have to be run the first time after installation


source ./setup.sh

[ ! -d "./Build" ] && mkdir Build
cd Build
cmake ../
make

# Moving the lib

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp ./source/libQPixRTD.so ../Library/.
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp ./source/libQPixRTD.dylib ../Library/.
fi

cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

# only this works for ubuntu
# cp ./source/lib* ../Library/.


