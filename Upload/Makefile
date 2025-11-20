CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -DNDEBUG
CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG -std=c++17
LDFLAGS = -fopenmp

all: baseline/sw_baseline optimized/sw_optimized

baseline/sw_baseline: baseline/sw_baseline.c
	$(CC) $(CFLAGS) -o baseline/sw_baseline baseline/sw_baseline.c -lm

optimized/sw_optimized: optimized/sw_optimized.cpp
	$(CXX) $(CXXFLAGS) -o optimized/sw_optimized optimized/sw_optimized.cpp $(LDFLAGS)

clean:
	rm -f baseline/sw_baseline optimized/sw_optimized

.PHONY: all clean
