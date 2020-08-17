cd ../Build/

rm -rf *

cmake ..

make

mv libQpix.a ../Library/.

cd ../root_read_test/

rm root_macro

make