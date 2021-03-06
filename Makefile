ARCH=$(shell uname -m)
ifeq ($(ARCH),i686)
        ARCHPREFIX=
	ARCHPOSTFIX=i686	
        MACHINE=-m32
else
        ARCHPREFIX=64
	ARCHPOSTFIX=x86_64
        MACHINE=-m64
endif

CUDAARCH=-arch sm_12
OMP=-fopenmp
XCOMPILER = -Xcompiler $(OMP)
HOMEDIR = $(CURDIR)
CU = /usr/local/cuda/bin/nvcc
CC = /usr/bin/gcc
CUDPPLIB = $(CUDPP_HOME)/lib
CUDPPINC = $(CUDPP_HOME)/cudpp/include/


INCLUDE = -I$(HOMEDIR) \
          -I$(CUDPPINC)

CUDALIBS = -lcudpp_$(ARCHPOSTFIX) -lcurand -lcudart -lcufft
LIBS  = -lconfig
harm : main.cu 
	$(CU) $(INCLUDE) $(CUDAARCH) -O2 -o harm  main.cu  -L$(CUDPPLIB) $(CUDALIBS)  $(LIBS)  $(XCOMPILER)

harm_debug : main.cu
	$(CU) $(INCLUDE) $(CUDAARCH) -g -o harm  main.cu -L$(CUDPPLIB) $(CUDALINS) $(LIBS)

df : df.c
	$(CC) $(INCLUDE) -O2 -o df df.c $(OMP)
