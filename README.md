# Q-Pix RTD signal simulation

Simulatin to get the response of qpix in LAr

It is set up so such that is requires no external dependencies. 

### Installing

After you clone the repo the setup.sh script should set up the enviorment and run cmake.
You will need to run this setup every time you launch a new terminal. 

```bash
bash setup.sh
```
This will make a build directory and run cmake and make, inorder to make a joint libary that will be moved into the Libary folder. 
Once this is done the toy examples can be tested

The toy examples are in test01 and test02




### test01

is the random number example can be ran form inside the test01 folder e.g.
```bash
cd test01
make test
./test
```

This generates some files that you can plot with the python notebook in the test01 directory, that show the distributions that have been generated. 


### test02
This can be copiled in the sme way as above and this produces a file of the diffused electron cloud.
The file it produces can be quite large since it is making an electron cloud and evey electrons position is written to the .txt file.
```bash
cd test02
make test
./test
```

### Example01
This run through all functions to produce a QPIX response. It takes an input file fom the geant4 step and outputs a file that is similar to an expected output with a pixel location and reset time. 

Once the enviorment is sourced the example can be made and ran by.
```bash
cd Example01
make Example01
./Example01
```
There is currently 3 example outputs from the geant4 of argon 39, muons, and protons. These are useful for some simple testing. 
