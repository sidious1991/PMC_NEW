#CC=g++ -m64
#LD=g++ -m64
CC=gcc -m64
LD=gcc -m64 
CFLAGS= -c -g -I . -ftree-vectorizer-verbose=1 -DLINUX -fpermissive -fopenmp
LDFLAGS= -lm -fopenmp

all: nnml 

nnml: nnml.o common.o iniparser.o dictionary.o 
	${LD} -o nnml nnml.o iniparser.o dictionary.o common.o ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o nnlm
