BLASDIR	= $(HOME)/magma-1.2.1
CUDADIR	= /opt/CUDA/4.1.28/cudatoolkit
TARGETS	= dgemm sgemm

#CC	= /usr/local/cuda/bin/nvcc
CC	= nvcc
#CXX	= /usr/bin/g++

OPTS	= -O2 #-fopenmp #-mavx
#OPTS	= -O2 -fopenmp -msse3
PROC_TYPE	= $(strip $(shell uname -m | grep 64))

CFLAGS	= -I../include -I$(BLASDIR)/include $(OPTIM) -I/usr/include/x86_64-linux-gnu -I.
LDFLAGS = -L$(BLASDIR)/lib -L$(CUDADIR)/lib64 -L. -lcuda -lcublas -lmagma -lmagmablas -L/usr/bin -lblas -lc

.PHONY: all clean test

all: $(TARGETS)

dgemm: gemm.c
	$(CC) -DTESTING_DOUBLE_PRECISION $(CFLAGS) $< $(LDFLAGS) -o $@
sgemm: gemm.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGETS)

test: all
	LD_LIBRARY_PATH=$(BLASDIR)/lib64:$(LD_LIBRARY_PATH) ./dgemm 0 0 0 1024 16 1
	LD_LIBRARY_PATH=$(BLASDIR)/lib64:$(LD_LIBRARY_PATH) ./sgemm 0 0 0 1024 16 1
