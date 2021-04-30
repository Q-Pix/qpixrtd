#!/bin/sh
printf '************************************* \n'
message="Hello human I will now set up your Qpix enviorment."; for ((i=0; i<${#message}; i++)); do echo "after 75" | tclsh; printf "${message:$i:1}"; done; echo;
sleep 1

# setting up the Qpix path
printf '************************************* \n'
export QPixRTDDir=$PWD
printf 'Defined your Qpix path as \n'
echo $QPixRTDDir
sleep 1


cd $QPixRTDDir

printf '************************************* \n'
printf 'Make/goto the Build folder and compile \n'
printf '\n'
sleep 1

# Making the build directory
mkdir Build
cd Build
cmake ../
make

sleep 1
printf '\n'
printf '************************************* \n'
printf 'Moving the joint library \n'

# Moving the lib
cp ./source/libQPixRTD.so ../Library/.
cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

cd $QPixRTDDir
printf '************************************* \n'
sleep 1

message="Human you may get to work now."; for ((i=0; i<${#message}; i++)); do echo "after 75" | tclsh; printf "${message:$i:1}"; done; echo;


