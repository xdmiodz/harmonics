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

HOMEDIR = $(CURDIR)
CU = /usr/local/cuda/bin/nvcc
CUDPPLIB = $(CUDPP_HOME)/lib
CUDPPINC = $(CUDPP_HOME)/cudpp/include/


INCLUDE = -I$(HOMEDIR) \
          -I$(CUDPPINC)

LIBS = -lcudpp_$(ARCHPOSTFIX) -lconfig -lcurand -lcudart -lcufft
harm : main.cu 
	$(CU) $(INCLUDE) $(CUDAARCH) -O2 -o harm  main.cu  -L$(CUDPPLIB) $(LIBS)

harm_debug : main.cu
	$(CU) $(INCLUDE) $(CUDAARCH) -g -o harm  main.cu -L$(CUDPPLIB) $(LIBS)
