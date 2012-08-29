BLASDIR	= /usr/local/cuda
TARGETS	= dgemm sgemm

CC	= LIBRARY_PATH=/usr/lib/x86_64-linux-gnu /usr/local/cuda/bin/nvcc
CXX	= /usr/bin/g++

OPTS	= -O2 #-fopenmp #-mavx
#OPTS	= -O2 -fopenmp -msse3
PROC_TYPE	= $(strip $(shell uname -m | grep 64))

CFLAGS	= -I../include -I$(BLASDIR)/include $(OPTIM) -I/usr/include/x86_64-linux-gnu
LDFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcublas -L/usr/bin -lblas -lc

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
