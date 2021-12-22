import unittest
from damavand.state import State
from damavand.circuit import Circuit


class TestOperation(unittest.TestCase):

    def test_Circuit(self):
        num_qubits = 2
        circuit = Circuit(num_qubits)
        circuit.add_hadamard_gate(0)
        return
