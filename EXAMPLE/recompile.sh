rm -rf ../Build/*

cd ../Build

cmake ..

make

rm -rf ../Library/*

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp ./source/libQPixRTD.so ../Library/.
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp ./source/libQPixRTD.dylib ../Library/.
fi

cp ./source/libQPixRTDDict_rdict.pcm ../Library/.
cp ./source/libQPixRTDDict.rootmap ../Library/.

cd ../EXAMPLE/

cd build
rm -rf ../build/*

cmake ..

make

cd ..

