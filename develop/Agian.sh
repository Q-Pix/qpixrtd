cd ../../Build/

make

cp ./source/libQPixRTD.so ../Library/.
cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

cd ../develop/build/

make clean 

make
