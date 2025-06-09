import threading
import time
import random
import signal
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication

class Neuron(threading.Thread):
    def __init__(self, network, neuron_id, position, threshold=1.0, decay_rate=0.01, initial_energy=10.0):
        super().__init__()
        self.network = network
        self.neuron_id = neuron_id
        self.position = position  # 3D position (x, y, z)
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.energy = initial_energy
        self.activation = 0.0
        self.alive = True
        self.lock = threading.Lock()
        self.connections = []  # List of connected neurons
        print(f"Neuron {self.neuron_id} initialized at position {self.position} with threshold {self.threshold}, decay_rate {self.decay_rate}, initial_energy {self.energy}")

    def receive_input(self, input_value):
        with self.lock:
            self.activation += input_value
            print(f"Neuron {self.neuron_id} at position {self.position} received input: {input_value}, new activation: {self.activation}")
            if self.activation >= self.threshold:
                self.fire()

    def fire(self):
        if self.energy > 0:
            print(f"Neuron {self.neuron_id} at position {self.position} fired with activation {self.activation} and energy {self.energy}")
            self.activation = 0.0
            self.energy -= 1  # Energy decreases with each firing
            print(f"Neuron {self.neuron_id} at position {self.position} energy after firing: {self.energy}")
            self.send_signal()
        if self.energy <= 0:
            self.die()

    def send_signal(self):
        for neuron in self.connections:
            signal_value = random.uniform(0.5, 1.5)
            print(f"Neuron {self.neuron_id} at position {self.position} sending signal {signal_value} to Neuron {neuron.neuron_id} at position {neuron.position}")
            neuron.receive_input(signal_value)

    def connect(self, other_neuron):
        self.connections.append(other_neuron)
        print(f"Neuron {self.neuron_id} at position {self.position} connected to Neuron {other_neuron.neuron_id} at position {other_neuron.position}")

    def die(self):
        if self.alive:
            print(f"Neuron {self.neuron_id} at position {self.position} has died.")
            self.alive = False

    def run(self):
        print(f"Neuron {self.neuron_id} at position {self.position} thread started.")
        while self.alive:
            with self.lock:
                self.energy -= self.decay_rate
                print(f"Neuron {self.neuron_id} at position {self.position} energy decayed to {self.energy}")
                if self.energy <= 0:
                    self.die()
            time.sleep(0.1)  # Simulate time passing
        print(f"Neuron {self.neuron_id} at position {self.position} thread ended.")


class NeuralNetwork:
    def __init__(self, num_neurons):
        self.neurons = [Neuron(self, neuron_id=i, position=(random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100))) for i in range(num_neurons)]
        for neuron in self.neurons:
            neuron.start()

    def connect_neurons(self, neuron1_index, neuron2_index):
        self.neurons[neuron1_index].connect(self.neurons[neuron2_index])

    def stop_all_neurons(self):
        for neuron in self.neurons:
            neuron.die()
        for neuron in self.neurons:
            neuron.join()

    def get_positions_and_states(self):
        positions = []
        states = []
        for neuron in self.neurons:
            positions.append(neuron.position)
            if neuron.alive:
                states.append('alive' if neuron.energy > 0 else 'dead')
            else:
                states.append('dead')
        return positions, states

    def get_connections(self):
        connections = []
        for neuron in self.neurons:
            for conn in neuron.connections:
                connections.append((neuron.position, conn.position))
        return connections


def run_visualization():
    app = QApplication(sys.argv)  # Initialize QApplication before creating any widgets

    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle('Neural Network Visualization')
    view.setCameraPosition(distance=200)

    grid = gl.GLGridItem()
    view.addItem(grid)

    scatter_plot = gl.GLScatterPlotItem()
    view.addItem(scatter_plot)

    connections_lines = gl.GLLinePlotItem()
    view.addItem(connections_lines)

    network = NeuralNetwork(num_neurons=5)  # Create NeuralNetwork after QApplication

    # Connect neurons in a simple chain
    for i in range(len(network.neurons) - 1):
        network.connect_neurons(i, i + 1)

    def update_visualization():
        positions, states = network.get_positions_and_states()
        positions_array = np.array(positions)
        colors = np.array([[0, 1, 0, 1] if state == 'alive' else [1, 0, 0, 1] for state in states])

        scatter_plot.setData(pos=positions_array, color=colors, size=5)

        connections = network.get_connections()
        if connections:
            lines = np.array([(start, end) for start, end in connections])
            lines = lines.reshape(-1, 3)
            connections_lines.setData(pos=lines, color=[0.5, 0.5, 0.5, 1])

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_visualization)
    timer.start(100)

    def signal_handler(sig, frame):
        print('Interrupt received, stopping...')
        network.stop_all_neurons()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while any(neuron.alive for neuron in network.neurons):
            random_neuron = random.choice(network.neurons)
            random_neuron.receive_input(random.uniform(0.5, 1.5))
            app.processEvents()
            time.sleep(random.uniform(0.5, 2))
    except KeyboardInterrupt:
        print('KeyboardInterrupt received, stopping...')
        network.stop_all_neurons()

    print("Neuron threads have finished.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_visualization()
