# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use
CC = gcc
NVCC = nvcc
EXECS =  v1 v2 v3 v0
.PHONY: $(EXECS)
all: $(EXECS)

v1:
	$(NVCC) src/v1.cu  -o v1
	./v1

v2:
	$(NVCC) src/v2.cu  -o v2
	./v2
v3:
	 $(NVCC) src/v3.cu  -o v3
	./v3
v0:
	$(CC) src/v0.c -o v0
	./v0
