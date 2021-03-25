#rm -rf ../Build/*

cd ../Build

cmake ..

make

mv libQpix.a ../Library/.

cd ../develop/

rm -rf build/*

cd build

cmake ..

make
