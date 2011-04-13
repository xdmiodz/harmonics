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

HOMEDIR = $(CURDIR)
CU = /usr/local/cuda/bin/nvcc
CUDPPLIB = $(CUDPP_HOME)/lib
CUDPPINC = $(CUDPP_HOME)/cudpp/include/

OMP=-fopenmp
XCOMPILER = -Xcompiler $(OMP)

INCLUDE = -I$(HOMEDIR) \
          -I$(CUDPPINC)

LIBS = -lcudpp_$(ARCHPOSTFIX) -lconfig -lcurand -lcudart -lgsl -lgslcblas -lm
harm : main.cu poisson1d.o
	$(CU) $(INCLUDE) -O2 -o harm  poisson1d.o main.cu  -L$(CUDPPLIB) $(LIBS) $(XCOMPILER)

harm_debug : main.cu poisson1d.o
	$(CU) $(INCLUDE) -g -o harm poisson1d.o main.cu -L$(CUDPPLIB) $(LIBS)

poisson1d.o : poisson1d.cu
	$(CU) $(INCLUDE) -c -o poisson1d.o poisson1d.cu