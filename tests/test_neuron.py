import unittest
import time
from evariste.neuron import Neuron, NeuralNetwork


class TestNeuron(unittest.TestCase):

    def setUp(self):
        self.network = NeuralNetwork(num_neurons=0)  # Empty network for testing
        self.neuron = Neuron(self.network, neuron_id=1, position=(1.0, 2.0, 3.0))

    def test_initialization(self):
        self.assertEqual(self.neuron.neuron_id, 1)
        self.assertEqual(self.neuron.position, (1.0, 2.0, 3.0))
        self.assertTrue(self.neuron.alive)
        self.assertEqual(self.neuron.activation, 0.0)

    def test_receive_input(self):
        initial_activation = self.neuron.activation
        self.neuron.receive_input(0.5)
        self.assertEqual(self.neuron.activation, initial_activation + 0.5)

    def test_connect_neurons(self):
        other_neuron = Neuron(self.network, neuron_id=2, position=(4.0, 5.0, 6.0))
        self.neuron.connect(other_neuron)
        self.assertIn(other_neuron, self.neuron.connections)

    def test_die(self):
        self.neuron.die()
        self.assertFalse(self.neuron.alive)


if __name__ == '__main__':
    unittest.main()
