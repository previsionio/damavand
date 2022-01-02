FROM rust

COPY . /repos/damavand
WORKDIR /repos/damavand

MAINTAINER Michel Nowak "michelnowak1@gmail.com"

# general
RUN apt update
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y python3
RUN apt install -y python3-pip

# damavand rust
RUN apt install -y autoconf
RUN apt install -y mpich
RUN apt install -y clang
RUN apt install -y libtool
RUN apt install -y texinfo

# damavand python
RUN pip install -r requirements-dev.txt
RUN pip install -r requirements-examples.txt
RUN python3 setup-cpu.py install
