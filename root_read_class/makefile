INCDIR = $(QpixDir)/include
LIBDIR = $(QpixDir)/Library

 
# Compiler flags
CFLAGS = -Wall -Wextra -Wno-long-long
CFLAGS += `root-config --cflags`
CFLAGS += -O3 -fno-common -c
CFLAGS += -I$(INCDIR)
CFLAGS += -std=c++17

LDFLAGS = -L$(LIBDIR) -lQpix
LDFLAGS += `root-config --glibs`
LDFLAGS += -Wl,-no_compact_unwind

root_macro: root_macro.cpp 
	$(CXX) $(CFLAGS) root_macro.cpp
	$(CXX) `root-config --cflags` -o root_macro root_macro.o $(LDFLAGS)
	rm root_macro.o

