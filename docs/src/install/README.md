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

1. install requirements
2. install from pypi
3. install from Dockerfile
4. setup development environment
5. install on supercomputers

## Requirements
The system requirements are already enumerated in `Dockerfiles/CPU/Dockerfile` and in `Dockerfiles/GPU/Dockerfile`. Make
sure that they are installed on you machine **before** installing from pypi. On Ubuntu, this would look like this:

### general
Make sure that python is installed along with its utility pip.
```bash
sudo apt install python3-pip
```

### compilation for damavand-gpu
Install compilators for the GPU library
```bash
sudo apt install g++
sudo apt install cmake
```

### damavand rust
Install Rust compilers.
```bash
sudo apt install curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install system dependencies.
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
## pypi fetch

Damavand is linked to pypi. Once a new version is tagged on github, pypi automatically updates the version to the
latest one.

If you wish to install damavand easily and with the GPU, simply use:

```bash
pip3 install damavand
```
On the other hand, if you do not have a CUDA-capable GPU on your machine, install as explained in
`Dockerfiles/CPU/Dockerfile`.

## Install via Dockerfile

If you wish to deploy damavand on a machine on which you do not want to install from source, you can use Dockerfiles, as
described here.

First, build the image in the root directory in CPU mode:

```bash
docker build -t damavand-cpu-image -f Dockerfiles/CPU/Dockerfile
```

or in GPU mode:

```bash
docker build -t damavand-gpu-image -f Dockerfiles/GPU/Dockerfile
```

Then, run the image with a bash prompt.
```bash
docker run -it damavand-gpu-image bash
```

This method simplifies the installation process, but does not provide with enough flexibility to run on multiple nodes,
as HPC architectures.


## Contributing to damavand development
Another mode of installation is by setting up the environement for development.

### Clone the repository
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
in debug mode: compile in release mode.

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
python3 setup.py install
```

## Install on supercomputers

In order to compile damavand locally, you will need to load some modules first.

```bash
module load rust
module load openmpi
module load automake
module load libtool
module load cmake
module load cuda
module load autoconf
module load llvm
module load gcc
```

Once the modules are loaded, you will be able to compile damavand, just as described in the previous sections.
However, you might lack an internet connection on the supercomputer, so the rust dependencies will not be downloaded.

In order to install damavand from sources, there is a workaround.

On your local computer, run:

```bash
cd damavand/
mkdir cargo
CARGO_HOME=$PWD/cargo cargo fetch
cd ..
zip -r damavand.zip damavand/
scp -r damavand.zip <user>@<supercomputer>:<path_to_working_directory>
```
