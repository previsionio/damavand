import numpy as np
import matplotlib.pylab as plt

qubits= np.genfromtxt("time_cpu/time_qubits.npy")
damavand_jean_zay = np.genfromtxt("time_cpu/time_damavand_results.npy")
damavand_gpu_jean_zay = np.genfromtxt("time_gpu/time_damavand_gpu_results.npy")
pennylane_jean_zay = np.genfromtxt("time_cpu/time_pennylane_results.npy")
qulacs_jean_zay = np.genfromtxt("time_cpu/time_qulacs_results.npy")

plt.plot(pennylane_jean_zay, color="blueviolet", label="pennylane")
plt.plot(qulacs_jean_zay, "--", color="blueviolet", label="qulacs")
plt.plot(damavand_jean_zay, color="lime", label="damavand-cpu")
plt.plot(damavand_gpu_jean_zay, "--", color="lime", label="damavand-gpu")

plt.ylabel("simulation time(s)")
plt.xlabel("number of qubits")
plt.legend(ncol=1, loc=2)
plt.yscale("log")
plt.savefig("time_benchmarks.png")
