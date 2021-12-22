import time
from tqdm import tqdm
import numpy as np
from damavand import Circuit
from mpi4py import MPI

def run_circuit(num_qubits, num_layers, apply_method="brute_force"):
    circuit = Circuit(num_qubits, apply_method=apply_method)
    
    for l in range(num_layers): 
        for i in range(num_qubits):
            circuit.add_rotation_x_gate(i, np.pi/(i+1))
    
    for i in range(num_qubits):
        circuit.add_pauli_z_gate(i, True)
    
    circuit.forward()
    samples = circuit.sample(20000)
    return np.mean(samples, axis=0)

apply_methods = ["multithreading"]

for apply_method in apply_methods:
    num_layers = 5
    times = []
    for num_qubits in range(6, 7):
        start = time.time()
        results = run_circuit(num_qubits, num_layers, apply_method)
        print(num_qubits)
        print(results)
        elapsed = time.time() - start
        times.append(elapsed)
        # np.savetxt("single_node.npy", times)
