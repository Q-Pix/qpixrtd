INCDIR = $(QpixDir)/include
LIBDIR = $(QpixDir)/Library

# Compiler flags
CFLAGS = -std=c++11 -Wall -Wextra -Wno-long-long \
	-O3 -fno-common -c \
	-I$(INCDIR)

LDFLAGS = -L$(LIBDIR) -lQpix

test: Electron_Diffusion.cc 
	$(CXX) $(CFLAGS) Electron_Diffusion.cc
	$(CXX)  -o Electron_Diffusion Electron_Diffusion.o $(LDFLAGS)
	rm Electron_Diffusion.o

