In order to use this executable, make a build directory and build the executable from the 

$ mkdir build
$ cd build
$ cmake ..
$ make

You can then run the executable by going back up into the RTD directory and running it like an executable

./RTD -input <input file> -output <output file> -threshold <reset threshold>

Other optional parameters include:
-noise
-recombination

Both parameters are automatically set to off (no noise, no recombination), but by adding -noise and -recombination to the end of your command,
you can turn one or both of these on.
