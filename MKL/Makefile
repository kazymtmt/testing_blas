BLASDIR	= /opt/intel/composer_xe_2011_sp1.10.319/mkl
TARGETS	= dgemm sgemm

CC	= gcc
CXX	= g++

OPTS	= -O2 -fopenmp -mavx
#OPTS	= -O2 -fopenmp -msse3

CFLAGS	= -I../include -I$(BLASDIR)/include  -std=c99 $(OPTS) -Wall
LDFLAGS = -fopenmp -L$(BLASDIR)/lib/intel_64 -lmkl_core -lmkl_gnu_thread -lm -lmkl_intel_lp64

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
