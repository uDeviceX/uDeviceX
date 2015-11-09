
test: main.o
	$(CXX) -fopenmp -O3 -march=native -mtune=native -o test main.o

main.o: main.cpp logistic.h timer.h dpd8x8.h
	$(CXX) main.cpp -std=c++11 -g -O3 -march=native -mtune=native -ffast-math -fopenmp -c -o main.o

dpd8x8.h: dpd8x8.mh
	m4 dpd8x8.mh > dpd8x8.h

