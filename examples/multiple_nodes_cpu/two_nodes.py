import time
import numpy as np
from damavand import Circuit
from mpi4py import MPI

def run_circuit(num_qubits, num_layers, apply_method="brute_force"):
    circuit = Circuit(num_qubits, apply_method=apply_method)
    
    for l in range(num_layers): 
        for i in range(num_qubits):
            circuit.add_rotation_x_gate(i, np.pi/(i+1))
    # circuit.add_rotation_x_gate(0, np.pi/2)
    # for i in range(num_qubits):
    #     circuit.add_rotation_x_gate(i, np.pi/4)
    
    for i in range(num_qubits):
        circuit.add_pauli_z_gate(i, True)
    
    circuit.forward()
    samples = circuit.sample()
    return np.mean(circuit.extract_expectation_values(samples), axis=0)

apply_methods = ["distributed_cpu"]

for apply_method in apply_methods:
    num_layers = 1
    times = []
    for num_qubits in range(3, 4):
        start = time.time()
        results = run_circuit(num_qubits, num_layers, apply_method)
        print(results)
        elapsed = time.time() - start
        times.append(elapsed)
        np.savetxt("two_nodes.npy", times)
