rm -rf ../Build/*

cd ../Build

cmake ..

make

mv libQpix.a ../Library/.

cd ../develop/

cd build

make clean

make
