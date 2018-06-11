CXX=g++
CXXFLAGS=-O3 -Wall
LDFLAGS=-lm
BIN=$(HOME)/bin

main: main.o com.o disp.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^


clean:; rm -rf *.o main
.PHONY: clean install

install: main
	mkdir -p $(BIN)
	cp main $(BIN)/u.post.rbc.flux
