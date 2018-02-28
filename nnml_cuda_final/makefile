CC=nvcc
LD=nvcc
CFLAGS=-arch=compute_30 -ccbin clang-3.8
LDFLAGS=-lm 

all: nnml_cuda

nnml_cuda: nnml_cuda.o common.o dictionary.o iniparser.o
	${LD} ${CFLAGS} nnml_cuda.o common.o dictionary.o iniparser.o -o nnml_cuda ${LDFLAGS}

nnml_cuda.o: nnml_cuda.cu
	$(CC) $(CFLAGS) -c nnml_cuda.cu $(LDFLAGS)

common.o: common.cu
	$(CC) $(CFLAGS) -c common.cu $(LDFLAGS)

dictionary.o: dictionary.cu
	$(CC) $(CFLAGS) -c dictionary.cu $(LDFLAGS)

iniparser.o: iniparser.cu
	$(CC) $(CFLAGS) -c iniparser.cu $(LDFLAGS)

clean:
	rm -rf *.o nnml_cuda
