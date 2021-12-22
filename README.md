# Damavand
![picture](figures/Damavand_in_winter.jpg)

Damavand is a code that simulates quantum circuits.
It is intended to simulate variational classifiers.

## Installation
Damavand is written in [Rust](https://www.rust-lang.org/). The recommended way to install Rust is
through [Rustup](https://rustup.rs/):
```
curl https://sh.rustup.rs -sSf | sh
```
It requires the nightly version to be installed. To do this, run:
```
rustup default nightly
```
## Dependencies
You will first need to install the dependencies of Damavand.

```
git clone --recurse-submodules https://github.com/MichelNowak1/damavand.git

cd damavand/
mkdir build/
cd build/
cmake ..
make
make install
```

## Building
Building in debug mode:
```
cargo build
```

Building in release mode:
```
cargo build --release
```


## Running
Running in debug mode:
```
cargo run --no-default-features 
```

Running in release mode:
```
cargo run --release --no-default-features 
```

## Runnning tests

You can run rust tests
```
cargo test --no-default-features
```

## Building documentation
You can create damavand's documentation with the following command:
```
cargo doc
```
It will end up in target/doc/damavand/index.html

You can create damavand's python bindings' documentation with the following commands:
```
cd doc && make html
```
