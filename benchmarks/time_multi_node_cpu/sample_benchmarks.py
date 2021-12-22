from damavand.utils import initialize_mpi
import time
from tqdm import tqdm
initialize_mpi()
import numpy as np
from damavand.qubit_backend.circuit import Circuit
import pennylane as qml


def run_circuit(num_qubits, target_qubit, apply_method="distributed_cpu"):
    circuit = Circuit(num_qubits, apply_method=apply_method)
    
    circuit.add_pauli_z_gate(target_qubit, True)
    
    circuit.forward()
    samples = circuit.sample()
    return np.mean(samples, axis=0)

apply_methods = ["distributed_cpu"]

for apply_method in apply_methods:
    num_qubits = 30
    times = []
    for target_qubit in range(0, 1):
        start = time.time()
        results = run_circuit(num_qubits, target_qubit, apply_method)
        elapsed = time.time() - start
        times.append(elapsed)
        print(target_qubit, elapsed)
        np.savetxt("time_damavand_results.npy", times)
