FROM ubuntu:21.04
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq install --yes --no-install-recommends \
g++ \
git \
libconfig-dev \
libhdf5-mpich-dev \
libmpich-dev \
make \
mpich \
nvidia-cuda-toolkit \
python3-dev
RUN echo root:g | chpasswd
SHELL ["/bin/bash", "-l", "-c"]
RUN echo 'PATH=$HOME/.local/bin:$PATH' >> $HOME/.profile
ENV GIT_SSL_NO_VERIFY=1

RUN git clone https://github.com/slitvinov/atest
RUN cd atest && make install

RUN git clone https://github.com/amlucas/bop
RUN cd bop && ./configure --prefix $HOME/.local
RUN cd bop && make MPICXX=mpicxx
RUN cd bop && make -s test
RUN cd bop && make install
RUN cd bop && make installconfig INST_BIN=$HOME/.local/bin

RUN git clone --quiet --single-branch --depth 1 https://github.com/uDeviceX/uDeviceX
RUN cd uDeviceX && make
RUN cd uDeviceX/src && ./configure
RUN cd uDeviceX/src && u.make -j4
