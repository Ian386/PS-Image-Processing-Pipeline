CC ?= gcc
MPICC ?= mpicc
CFLAGS := -O3 -Wall -Wextra -std=c11 -Ikernels
LDFLAGS := -lm

KDIR := kernels
BDIR := build

COMMON_SRC := $(KDIR)/filters_seq.c $(KDIR)/image_io.c $(KDIR)/bench.c
COMMON_OBJ := $(BDIR)/filters_seq.o $(BDIR)/image_io.o $(BDIR)/bench.o

.PHONY: all seq omp pthread mpi clean dirs

all: seq

dirs:
	@mkdir -p $(BDIR) results data/reference

$(BDIR)/%.o: $(KDIR)/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

seq: dirs $(COMMON_OBJ) $(BDIR)/main_seq.o
	$(CC) $(CFLAGS) -o $(BDIR)/seq $(BDIR)/main_seq.o $(COMMON_OBJ) $(LDFLAGS)
	@echo "Built $(BDIR)/seq"

omp: dirs $(COMMON_OBJ) $(BDIR)/filters_omp.o $(BDIR)/main_omp.o
	$(CC) $(CFLAGS) -fopenmp -o $(BDIR)/omp $(BDIR)/main_omp.o $(BDIR)/filters_omp.o $(COMMON_OBJ) $(LDFLAGS)
	@echo "Built $(BDIR)/omp"

$(BDIR)/filters_omp.o: $(KDIR)/filters_omp.c | dirs
	$(CC) $(CFLAGS) -fopenmp -c -o $@ $<

$(BDIR)/main_omp.o: $(KDIR)/main_omp.c | dirs
	$(CC) $(CFLAGS) -fopenmp -c -o $@ $<

pthread: dirs $(COMMON_OBJ) $(BDIR)/thread_pool.o $(BDIR)/filters_pthread.o $(BDIR)/main_pthread.o
	$(CC) $(CFLAGS) -pthread -o $(BDIR)/pthread $(BDIR)/main_pthread.o $(BDIR)/filters_pthread.o $(BDIR)/thread_pool.o $(COMMON_OBJ) $(LDFLAGS)
	@echo "Built $(BDIR)/pthread"

mpi: dirs $(COMMON_OBJ) $(BDIR)/filters_mpi.o $(BDIR)/main_mpi.o
	$(MPICC) $(CFLAGS) -o $(BDIR)/mpi_pipeline $(BDIR)/main_mpi.o $(BDIR)/filters_mpi.o $(COMMON_OBJ) $(LDFLAGS)
	@echo "Built $(BDIR)/mpi_pipeline"

$(BDIR)/filters_mpi.o: $(KDIR)/filters_mpi.c | dirs
	$(MPICC) $(CFLAGS) -c -o $@ $<

$(BDIR)/main_mpi.o: $(KDIR)/main_mpi.c | dirs
	$(MPICC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BDIR)
