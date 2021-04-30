rm -rf ../Build/*

cd ../Build

cmake ..

make

cp ./source/libQPixRTD.so ../Library/.
cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

cd ../develop/

rm -rf build/*

cd build

cmake ..

make
