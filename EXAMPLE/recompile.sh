cd ../Build/

rm -rf *

cmake ..

make

mv libQpix.a ../Library/.

cd ../EXAMPLE/

rm EXAMPLE

make