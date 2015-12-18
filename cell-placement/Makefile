all: pachinko cell-placement

CXXFLAGS = -O3 -std=c++11

pachinko: pachinko.o checker.o
	$(CXX) $^ -o $@ $(CXXFLAGS)

cell-placement:	main.o checker.o
	$(CXX) $^ -o $@ $(CXXFLAGS) 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f cell-placement pachinko *.o

.PHONY = clean
