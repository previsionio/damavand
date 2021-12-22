import numpy as np
import matplotlib.pylab as plt

qubits= np.genfromtxt("time_qubits.npy")
damavand_jean_zay = np.genfromtxt("time_damavand_results.npy")
pennylane_jean_zay = np.genfromtxt("time_pennylane_results.npy")
qulacs_jean_zay = np.genfromtxt("time_qulacs_results.npy")

plt.plot(qubits[1:], pennylane_jean_zay[1:], color="blueviolet", label="pennylane")
plt.plot(qubits[1:], qulacs_jean_zay[1:], "--", color="blueviolet", label="qulacs")
plt.plot(qubits[1:], damavand_jean_zay[1:], color="lime", label="damavand")

plt.ylabel("simulation time(s)")
plt.xlabel("number of qubits")
plt.legend(ncol=1, loc=2)
plt.yscale("log")
plt.savefig("time_benchmarks.png")
