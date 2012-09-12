#BLASDIR	= /usr/local/cuda-4.2
BLASDIR	= /usr/local/cuda-5.0
TARGETS	= dgemm sgemm

CC	= /usr/local/cuda/bin/nvcc
#CXX	= /usr/bin/g++

OPTS	= -O2 #-fopenmp #-mavx
#OPTS	= -O2 -fopenmp -msse3
PROC_TYPE	= $(strip $(shell uname -m | grep 64))

CFLAGS	= -I../include -I$(BLASDIR)/include $(OPTIM) -I/usr/include/x86_64-linux-gnu
LDFLAGS = -L$(BLASDIR)/lib64 -lcuda -lcublas -L/usr/bin -lblas -lc

.PHONY: all clean test

all: $(TARGETS)

dgemm: gemm.c
	$(CC) -DTESTING_DOUBLE_PRECISION $(CFLAGS) $< $(LDFLAGS) -o $@
sgemm: gemm.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGETS)

test: all
	LD_LIBRARY_PATH=$(BLASDIR)/lib64 ./dgemm 0 0 0 256 16 1
	LD_LIBRARY_PATH=$(BLASDIR)/lib64 ./sgemm 0 0 0 256 16 1
