cd ../Build/

rm -rf *

cmake ..

make

mv libQpix.a ../Library/.

cd ../root_read_class/

rm root_macro

make