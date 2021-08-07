FROM ubuntu:21.04
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq install --yes --no-install-recommends \
g++ \
git \
make \
pkg-config
RUN echo root:g | chpasswd
SHELL ["/bin/bash", "-l", "-c"]
RUN GIT_SSL_NO_VERIFY=1 git clone --quiet --single-branch --depth 1 https://github.com/uDeviceX/uDeviceX
RUN echo 'PATH=$HOME/.local/bin:$PATH' >> $HOME/.profile
RUN cd corpuscles && MAKEFLAGS=-j4 ./install.sh
RUN cd corpuscles/example/hello && make && ./main
