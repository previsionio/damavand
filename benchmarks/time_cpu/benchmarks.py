from damavand.utils import initialize_mpi
import time
from tqdm import tqdm
initialize_mpi()
import numpy as np
from damavand.qubit_backend.circuit import Circuit
import pennylane as qml



def run_pennylane(num_qubits, num_layers, device="lightning.qubit", weights=None):
    start = time.time()
    dev = qml.device(device, wires=num_qubits, shots=1000)
    @qml.qnode(dev, interface="autograd")
    def circuit(weights=None):
        for l in range(num_layers):
            for i in range(num_qubits):
                qml.RX(np.pi/(i+1), wires=i)
        # return qml.state()
        return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

    result = circuit()
    print(result)
    stop = time.time()
    return result, stop - start
def run_circuit(num_qubits, num_layers, apply_method="multithreading"):
    circuit = Circuit(num_qubits, apply_method=apply_method)
    
    for l in range(num_layers):
        for i in range(num_qubits):
            circuit.add_rotation_x_gate(i, np.pi/(i+1))
    
    for i in range(num_qubits):
        circuit.add_pauli_z_gate(i, True)
    
    circuit.forward()
    samples = circuit.sample()
    return np.mean(samples, axis=0)

apply_methods = ["multithreading"]

for apply_method in apply_methods:
    num_layers = 10
    max_qubits = 31
    qubits = []
    times = []
    for num_qubits in range(2, max_qubits):
        start = time.time()
        results = run_circuit(num_qubits, num_layers, apply_method)
        print(num_qubits, results)
        elapsed = time.time() - start
        times.append(elapsed)
        qubits.append(num_qubits)
        np.savetxt("time_damavand_qubits.npy", qubits)
        np.savetxt("time_damavand_results.npy", times)

    qubits = []
    times = []
    for num_qubits in range(2, max_qubits):
        start = time.time()
        results = run_pennylane(num_qubits, num_layers, "lightning.qubit")
        print(num_qubits, results)
        elapsed = time.time() - start
        times.append(elapsed)
        qubits.append(num_qubits)
        np.savetxt("time_pennylane_qubits.npy", qubits)
        np.savetxt("time_pennylane_results.npy", times)

    qubits = []
    times = []
    for num_qubits in range(2, max_qubits):
        start = time.time()
        results = run_pennylane(num_qubits, num_layers, "qulacs.simulator")
        print(num_qubits, results)
        elapsed = time.time() - start
        times.append(elapsed)
        qubits.append(num_qubits)
        np.savetxt("time_qulacs_qubits.npy", qubits)
        np.savetxt("time_qulacs_results.npy", times)
