# -*- makefile -*-

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
GCC_FLAGS1 = -fPIC -Wl,-Bsymbolic-functions -c -O3
GCC_FLAGS2 = -shared -O3 -Wl,-Bsymbolic-functions,-soname,tidal.so
endif
ifeq ($(UNAME_S),Darwin)
GCC_FLAGS1 = -fPIC -c
GCC_FLAGS2 = -shared -Wl,-install_name,tidal.so
endif

GCC = gcc

.PHONY: all
.SILENT: all

all:
	echo "Compiling C source code..."
	${GCC} ${GCC_FLAGS1} tidal.c
	echo "Generating shared library..."
	gcc ${GCC_FLAGS2} -o tidal.so tidal.o -lc
	rm tidal.o
	echo "Install successful."