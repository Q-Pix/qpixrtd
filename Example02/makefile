INCDIR = $(QpixDir)/include
LIBDIR = $(QpixDir)/Library

# Compiler flags
CFLAGS = -std=c++11 -Wall -Wextra -Wno-long-long \
	-O3 -fno-common -c \
	-I$(INCDIR)

LDFLAGS = -L$(LIBDIR) -lQpix

test: Example02.cc 
	$(CXX) $(CFLAGS) Example02.cc
	$(CXX)  -o Example02 Example02.o $(LDFLAGS)
	rm Example02.o

