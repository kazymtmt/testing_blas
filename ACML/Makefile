BLASDIR	= /opt/acml5.1.0/gfortran64_fma4_mp
#BLASDIR	= /opt/acml4.4.0/gfortran64
TARGETS	= dgemm sgemm

CC	= gcc
CXX	= g++

OPTS	= -O2 -fopenmp -mfma4
#OPTS	= -O2 #-fopenmp #-mavx
#OPTS	= -O2 -fopenmp -msse3

PROC_TYPE	= $(strip $(shell uname -m | grep 64))
CFLAGS	= -I../include -I$(BLASDIR)/include  -std=c99 $(OPTS) -Wall
LDFLAGS = -lgfortran -L$(BLASDIR)/lib -lacml_mp -fopenmp -lblas

## Open64
#CC	= /opt/open64-5.0/bin/opencc
#CC	= /opt/x86_open64-4.5.1/bin/opencc
#OPTS	= -O2 -mp -mfma4
#CFLAGS	= -I$(BLASDIR)/include  -std=c99 $(OPTS) -Wall
#LDFLAGS = -L/opt/open64-5.0/lib/gcc-lib/x86_64-open64-linux/5.0/64 -L/opt/open64-5.0/lib/gcc-lib/x86_64-open64-linux/5.0/64 -lfortran -L$(BLASDIR)/lib -lacml_mp -lffio -mp

.PHONY: all clean test

all: $(TARGETS)

dgemm: gemm.c
	$(CC) -DTESTING_DOUBLE_PRECISION $(CFLAGS) $< $(LDFLAGS) -o $@
sgemm: gemm.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGETS)

test: all
	./dgemm 0 0 0 256 16 1
	./sgemm 0 0 0 256 16 1
