import unittest
from damavand.state import State

class TestState(unittest.TestCase):
    def test_load_state(self):
        num_qubits=2
        state = State(num_qubits)
        return
