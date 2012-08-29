#BLASDIR	= /opt/clAmdBlas-1.6.236
#BLASDIR	= /opt/clAmdBlas-1.7.257
BLASDIR	= /opt/clAmdBlas-1.8.269
LIBLABEL = clAmdBlas
TARGETS = dgemm sgemm

CC	= gcc
CXX	= g++

OPTS	= -O2 #-fopenmp #-mavx
#OPTS	= -O2 -fopenmp -msse3

ifndef AMDAPPSDKROOT
	AMDAPPSDKROOT = /opt/AMDAPP
endif
OPENCLSDKROOT = $(AMDAPPSDKROOT)
CFLAGS	= -I$(BLASDIR)/include -I../include -I$(OPENCLSDKROOT)/include -std=c99 $(OPTS) -Wall
LDFLAGS = -lOpenCL -lblas -fopenmp -L$(BLASDIR)/lib64 -lclAmdBlas

PROC_TYPE	= $(strip $(shell uname -m | grep 64))
ifeq ($(PROC_TYPE),)
	LDFLAGS	+= -m32 -L$(OPENCLSDKROOT)/lib/x86
else
	LDFLAGS	+= -m64 -L$(OPENCLSDKROOT)/lib/x86_64
endif

.PHONY: all clean test

all: $(TARGETS)

dgemm: gemm.c
	$(CC) -DTESTING_DOUBLE_PRECISION $(CFLAGS) $< $(LDFLAGS) -o $@
sgemm: gemm.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGETS)

test: all
	./dgemm 0 0 0 128 16 1
	./sgemm 0 0 0 128 16 1
