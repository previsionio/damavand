# Damavand

Damavand is a code that simulates quantum circuits.
In order to learn more about damavand, refer to the [documentation](https://michelnowak1.github.io/damavand/).

## Development status

1. Core features:
  * single node CPU ✔️
  * single node GPU ✔️
  * multiple nodes CPU ✔️
  * single node multiple GPUs ✔️
  * multiple nodes multiple GPUs ✔️
2. Documentation 🛠️ (under construction) 
3. README.md 🛠️ (under construction) 
4. pypi 🛠️ (under construction) 

## Installation

### From pypi
```bash
pip3 install damavand
```

### From sources

```bash
git clone https://github.com/MichelNowak1/damavand.git
cd damavand/
python3 setup.py install
```

## Example

```python
from damavand import Circuit

# initialize MPI
from mpi4py import MPI

num_qubits = 1
circuit = Circuit(num_qubits)

circuit.add_hadamard_gate(0)

circuit.forward()
circuit.measure()

num_samples=10
circuit.sample(num_samples)
```

![picture](figures/Damavand_in_winter.jpg)
