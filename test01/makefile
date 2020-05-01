INCDIR = $(QpixDir)/include
LIBDIR = $(QpixDir)/Library

# Compiler flags
CFLAGS = -Wall -Wextra -Wno-long-long \
	-O3 -fno-common -c \
	-I$(INCDIR)

LDFLAGS = -L$(LIBDIR) -lQpix

test: test.cc 
	$(CXX) $(CFLAGS) test.cc
	$(CXX)  -o test test.o $(LDFLAGS)
	rm test.o

