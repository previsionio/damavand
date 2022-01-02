FROM nvidia/cuda:11.0-devel

COPY . /repos/damavand
WORKDIR /repos/damavand

MAINTAINER Michel Nowak "michelnowak1@gmail.com"

# general
RUN apt update
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y python3
RUN apt install -y python3-pip

# compilation for damavand
RUN apt install -y g++

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt install -y cmake

# damavand rust
RUN apt install -y autoconf
RUN apt install -y mpich
RUN apt install -y clang
RUN apt install -y libtool
RUN apt install -y texinfo

# install rust
RUN apt install -y \
    build-essential \
    curl
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"


# CUDA samples
RUN cd /repos && git clone https://github.com/NVIDIA/cuda-samples.git
ENV CUDA_SAMPLES_COMMON_INCLUDE_DIR="/repos/cuda-samples/Common/"

# damavand python
RUN pip install -r requirements-dev.txt
RUN pip install -r requirements-examples.txt

# damavand GPU
RUN pip install damavand
RUN mkdir /repos/damavand/damavand-gpu/build
RUN cd /repos/damavand/damavand-gpu/build && \
    cmake .. && \
    make

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/repos/damavand/damavand-gpu/build"
