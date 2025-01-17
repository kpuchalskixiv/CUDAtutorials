
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

all:    prac2 avg

NVCCFLAGS	:= -lineinfo --ptxas-options=-v --use_fast_math

prac2:	prac2.cu Makefile
	nvcc prac2.cu -o prac2 $(INC) $(NVCCFLAGS) $(LIB)

prac2_device:	prac2_device.cu Makefile
	nvcc prac2_device.cu -o prac2_device $(INC) $(NVCCFLAGS) $(LIB)

avg:  avg.cu Makefile
	nvcc avg.cu -o avg $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2 avg
