This is the example file, currently set up to simulate the detector response 
of a given Geant4 simulated event. It will simulate the propogation and
diffusion, and recombination of electrons from the original tracks to the 
detector wall.

To make the EXAMPLE excutable, follow the following steps from the EXAMPLE
directory:

```
cd build
cmake ..
make
```

and to run the example, follow the following steps from the EXAMPLE directory:

```
./build/EXAMPLE /path/to/input_file.root /path/to/output_file.root
```

where it will create a new output_file.root in the location you provided.
