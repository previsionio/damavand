---
sidebarDepth: 3
---

# Pennylane plugin

Damavand enters the quantum simulators community after many others. Among the most popular quantum circuit simulators
we can cite [PennyLane](https://pennylane.ai/): a python library developped by the canadian company [Xanadu](https://xanadu.ai)
specialized in photonics hardware. PennyLane integrates gradient descent routines and many other features that makes it
the most popular and efficient library to push research in the field of **Quantum Machine Learning**.

It was thus legitimate to integrate damavand to PennyLane through a plugin: **pennylane-damavand**.
This page shows presents how damavand can be leveraged on supercomputers with all the functionalities of pennylane.

## Gradient Descent
One of the main features of PennyLane is to integrate gradient descent schemes. 

<p align="center">
  <img src="./damavand_gradient_descent.png" width="400em" />
</p>


```python
num_qubits = 10
num_layers = 5

# initialize device with damavand.qubit backend
dev = qml.device("damavand.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit():
    # build some layers
    for l in range(num_layers): 
        qml.RX(np.pi/(i+1), wires=0)
    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

result = circuit()
```
