CC=gcc
LD=gcc

MKL=0
OPENBLAS=1

MKLDIR=/opt/intel
BLASDIR=/opt/OpenBLAS

OPT_LEVEL=-flto -Ofast -fomit-frame-pointer

CC_OPTS=$(OPT_LEVEL) -fPIC -Wall -Wno-unused-variable -Werror -Wfatal-errors
LD_OPTS=$(OPT_LEVEL) -lm -lc -lpthread

ifeq ($(MKL), 1)
  CC_OPTS:=$(CC_OPTS) -I$(MKLDIR)/mkl/include -DUSE_MKL
  ifeq ($(OS), Darwin)
    LD_OPTS:=$(LD_OPTS) -L${MKLDIR}/lib -L${MKLDIR}/mkl/lib/intel64 -liomp5 -lmkl_rt
  else
    LD_OPTS:=$(LD_OPTS) -L${MKLDIR}/lib -L${MKLDIR}/mkl/lib -liomp5 -lmkl_rt
  endif
endif

ifeq ($(OPENBLAS), 1)
  CC_OPTS:=$(CC_OPTS) -I$(BLASDIR)/include
  LD_OPTS:=$(LD_OPTS) -L$(BLASDIR)/lib -lopenblas
endif

HEADERS:=$(wildcard *.h) Makefile

.SUFFIXES:

TARGETS=main

all : $(TARGETS)

#%.so: %.o mmul.o rand.o mat.o $(DEPS)
#	$(CC) $(CC_OPTS) $^ -shared -o $@ $(LD_OPTS)
#
#%: %.o mmul.o rand.o utils.o $(DEPS)
#	$(CC) $(CC_OPTS) $^ -o $@ $(LD_OPTS)

%: %.o rand.o mat.o $(DEPS)
	$(CC) $(CC_OPTS) $^ -o $@ $(LD_OPTS)

%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@ $(CC_OPTS)

clean:
	rm -rf $(TARGETS) *.o
