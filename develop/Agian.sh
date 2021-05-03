cd ../../Build/

make

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp ./source/libQPixRTD.so ../Library/.
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp ./source/libQPixRTD.dylib ../Library/.
fi

cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

cd ../develop/build/

make clean 

make
