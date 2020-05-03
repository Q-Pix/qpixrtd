INCDIR = $(QpixDir)/include
LIBDIR = $(QpixDir)/Library

# Compiler flags
CFLAGS = -std=c++11 -Wall -Wextra -Wno-long-long \
	-O3 -fno-common -c \
	-I$(INCDIR)

LDFLAGS = -L$(LIBDIR) -lQpix

test: Example01.cc 
	$(CXX) $(CFLAGS) Example01.cc
	$(CXX)  -o Example01 Example01.o $(LDFLAGS)
	rm Example01.o

