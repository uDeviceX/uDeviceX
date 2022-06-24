M=../make
S=src
BOP=../src

include $M/usr.mk
include $M/common.mk
include $M/rules.mk

PYBIND_INC=$(PYBIND_PREFIX)/include
PYCONF=python3-config

BOP_CXXFLAGS = -I$(BOP)
BOP_LDFLAGS  = -L$(BOP)
LIBS         = -lbop -lbop_serial

PYBIND_CXXFLAGS = -I$(PYBIND_INC)
PYBIND_LDFLAGS  =

PY_CXXFLAGS = `$(PYCONF) --includes`
PY_LDFLAGS  = `$(PYCONF) --ldflags`

INCFLAGS = $(BOP_CXXFLAGS)  \
	 $(PYBIND_CXXFLAGS) \
	 $(PY_CXXFLAGS)     \
	 -fvisibility=hidden \
	-Wno-deprecated

LDFLAGS = $(BOP_LDFLAGS)  \
	$(PYBIND_LDFLAGS) \
	$(PY_LDFLAGS)

PYEXT  = $(shell $(PYCONF) --extension-suffix)
TARGET = pybop$(PYEXT)

all: $(TARGET)

$S/pybop.o: $S/pybop.cpp $S/pybop.h
$S/binding.o: $S/binding.cpp $S/pybop.h

$(TARGET): $S/binding.o $S/pybop.o
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS) $(LIBS)

clean:; rm -f $S/*.o $(TARGET)

test: $(TARGET)
	atest `find test -name main`

install: $(TARGET)

.PHONY: test clean test install
.SUFFIXES:
