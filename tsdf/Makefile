CPPFLAGS = -O2 -g -Wall -Wextra

all: sdf2volume sdf2volume2 sdf2vtk mergesdf

BIN  = $(HOME)/bin
CONF = $(HOME)/.udx

install: install_bin install_conf

install_bin: ;  cp tsdf tsdf.awk sdf2volume sdf2volume2 sdf2vtk mergesdf    $(BIN)
install_conf: ; mkdir -p $(CONF) && \
	        cp processor.tmp.cpp                               $(CONF)

clean: ; -rm sdf2volume sdf2vtk mergesdf
