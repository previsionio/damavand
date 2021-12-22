
# general
RUN sudo apt install vim
RUN sudo apt install git
RUN sudo apt install python3-pip

# compilation for damavand-gpu
RUN sudo apt install g++
RUN sudo apt install cmake

# install rust
RUN sudo apt install curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# damavand rust
RUN sudo apt install autoconf
RUN sudo apt install mpich
RUN sudo apt install clang
RUN sudo apt install libtool
RUN sudo apt install texinfo

# damavand python
RUN pip3 install setuptools-rust
RUN pip3 install matplotlib
RUN pip3 install pennylane
