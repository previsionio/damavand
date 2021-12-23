---
sidebarDepth: 2
---

# Installation

**Damavand**'s core is written in [Rust](https://www.rust-lang.org/fr). It is binded with a C++ library: **damavand-gpu** that
allows to take full advantage of **Nvidia GPUs** through the [CUDA framework](https://developer.nvidia.com/cuda-zone).
Finally, damavand is wrapped in **Python**.

<p align="center">
  <img src="/damavand/rust_logo.png" width="100em" />
  <img src="/damavand/cpp_logo.png" width="100em" /> 
  <img src="/damavand/cuda_logo.png" width="100em" /> 
  <img src="/damavand/python_logo.png" width="100em" />
</p>

So there are different ways to install damavand on your machine.

1. install only for Python usage
2. install from sources
2. install from Dockerfile

This page will guide you through the 3 different possibilities.


## Easy-Install

Damavand is linked to pypi. Once a new version is tagged on github, pypi automatically updates the version to the
latest one.

If you wish to install damavand easily, simply use:

```bash
pip3 install damavand
```
For further information on the usage of damavand, visit the next page of this documentation: [guide](/guide).

## Requirements
The system requirements are already enumerated in the Dockerfile. Make
sure that they are installed on you machine. On Ubuntu, this would look like this:

### general
```bash
sudo apt install python3-pip
```

### compilation for damavand-gpu
```bash
sudo apt install g++
sudo apt install cmake
```

### install rust
```bash
sudo apt install curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### damavand rust
```bash
sudo apt install autoconf
sudo apt install mpich
sudo apt install clang
sudo apt install libtool
sudo apt install texinfo
```

### damavand python
```bash
pip3 install setuptools-rust
pip3 install matplotlib
pip3 install pennylane
```
## Install from sources

### Cloning the repository
If you wish to develope further functionalities or correct bugs in damavand, you must first clone the git repository:

```bash
git clone https://github.com/previsionio/damavand.git
cd damavand/
```

### Building
Then comes the compiling phase, which can be done in two different modes.

If you install damavand in **debug** mode, you will be able to track the errors thanks to the provided traceback.
Simply run:

```bash
cargo build
```

Once you are confident that the code is functional, you can build damavand in release mode, which will run faster than
in debug mode.

```bash
cargo build --release
```

You can also install damavand as a Python library directly
First, install requirements:

```bash
pip3 install -r requirements-dev.txt
```

Then, execute:
```bash
python3 setup.py develop
```
in debug mode, or 

```bash
python3 setup.py install
```
in release mode.

## Install via Dockerfile

If you wish to deploy damavand on a machine on which you do not want to install from source, you can use Dockerfiles, as
described here.

First, build the image in the root directory:
```bash
docker build -t damavand-image .
```

Then, run the image with a bash prompt.
```bash
docker run -it damavand-image bash
```

This method simplifies the installation process, but does not provide with enough flexibility to run on multiple nodes,
as HPC architectures.


