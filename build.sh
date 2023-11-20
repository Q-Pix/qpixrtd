#!/bin/sh

###########################################
# build.sh
#
# Author: Dave Elofson
# Created: 2022-03-31
# This script will make the build directory, then run cmake and make. Should
# only have to be run the first time after installation.
#
# run by using the following command:
# ./build.sh
#
##########################################

initdir=$(pwd)

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

echo Building the EXAMPLE...;
echo $(pwd -P);
cd $initdir/EXAMPLE;
[ -d "build" ] || mkdir build;
cd build;
cmake ..;
make;
cd $initdir;

echo Building RTD...;
cd $initdir/RTD;
[ -d "build" ] || mkdir build;
cd build;
cmake ../;
make;
cd $initdir;

