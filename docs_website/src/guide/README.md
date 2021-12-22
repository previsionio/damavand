---
sidebarDepth: 2
---

# Guide

This page intends to present the functioning of **damavand** in a local mode. The next page HPC is dedicated to the High
Performance Computing mode.

## State of the art
Quantum cirucit simulation is already a rich domain where different methods have been proposed to efficiently simulate
the behavior of quantum systems. In the remainder of this page, we focus our attention on the simulation of pure quantum
states in a circuit based on the **qubit** model.

### State vector
Damavand is a state vector simulator, it allows to update the description of a quantum state gate after gate through the
circuit. The finally updated state can be measured. There are several ways to update the state vector. Here, we discribe
the different implementations, by referring to them as **apply methods**.

### Tensor networks
Tensor networks are a more recent alternative to state vector simulations. It is not implemented in damavand **yet**.

## Apply methods
Let us enumerate the 3 different apply methods implemented in damavand.

### Brute Force
The most natural way to update a quantum state after applying some operation on it is to build the unitary matrix that
shifts the quantum state from 0 to something else. If the system is composed of N qubits, then we need to build a matrix
of size 2^{2N} and apply it to the state before applying the gate in question.

Consider that we wish to apply a single qubit gate to the qubit k. Then, the unitary matrix, which we refer to as to the
**full matrix** is built from the tensor product:

F = I x ... x I x G x I x ... x I

<p align="center">
  <h4> Target qubit = 0 </h4>
  <img src="/IIU.png" width="500em" /> 
  <h4> Target qubit = 1 </h4>
  <img src="/IUI.png" width="500em" /> 
  <h4> Target qubit = 2 </h4>
  <img src="/UII.png" width="500em" />
</p>

where I is the 2x2 identity matrix and G represents the matrix of the gate of interest.
This means that in addition to storing the state vector that is of size 2^N, we need to build and store the matrix F.
This requires a large amount of memory and will further limit our capacity of simulating large quantum systems.

In the following example, we build a circuit of 10 qubits, and update the state vector with the **"brute_force"** apply
method.
```python
import damavand as dvd

num_qubits = 10
circuit = Circuit(10, apply_method="brute_force")
circuit.add_rotation_x(5)
circuit.forward()
```

### Shuffle
```python
import damavand as dvd

num_qubits = 10
circuit = dvd.Circuit(10, apply_method="shuffle")
circuit.add_rotation_x(5)
circuit.forward()
```

### Multithreading
```python
import damavand as dvd

num_qubits = 10
circuit = dvd.Circuit(10, apply_method="multithreading")
circuit.add_rotation_x(5)
circuit.forward()
```
