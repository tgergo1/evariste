import numpy as np
import random
import networkx as nx
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from evariste.neuron import Neuron
from evariste.models.synapses import Synapse

class Network:
    def __init__(self, name: str = "default_network"):
        self.name = name
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: List[Synapse] = []
        self.input_layer: List[str] = []
        self.output_layer: List[str] = []
        self.hidden_layers: List[List[str]] = []

    def add_neuron(self, neuron: Neuron, layer_type: str = "hidden", layer_index: int = 0) -> None:
        if neuron.id in self.neurons:
            raise ValueError(f"Neuron with ID {neuron.id} already exists in the network")

        self.neurons[neuron.id] = neuron

        if layer_type == "input":
            self.input_layer.append(neuron.id)
        elif layer_type == "output":
            self.output_layer.append(neuron.id)
        elif layer_type == "hidden":
            while len(self.hidden_layers) <= layer_index:
                self.hidden_layers.append([])
            self.hidden_layers[layer_index].append(neuron.id)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def connect(self, source_id: str, target_id: str, weight: float = 1.0) -> Synapse:
        if source_id not in self.neurons:
            raise ValueError(f"Source neuron {source_id} not found in network")
        if target_id not in self.neurons:
            raise ValueError(f"Target neuron {target_id} not found in network")

        source = self.neurons[source_id]
        target = self.neurons[target_id]

        synapse = Synapse(source, target, weight)
        self.synapses.append(synapse)
        return synapse

    def get_neuron(self, neuron_id: str) -> Optional[Neuron]:
        return self.neurons.get(neuron_id)

    def forward(self, inputs: Dict[str, float]) -> Dict[str, float]:
        for neuron in self.neurons.values():
            neuron.reset()

        for neuron_id, value in inputs.items():
            if neuron_id in self.neurons:
                self.neurons[neuron_id].set_input(value)
            else:
                raise ValueError(f"Input neuron {neuron_id} not found in network")

        self._process_layer(self.input_layer)
        for hidden_layer in self.hidden_layers:
            self._process_layer(hidden_layer)
        self._process_layer(self.output_layer)

        outputs = {}
        for neuron_id in self.output_layer:
            outputs[neuron_id] = self.neurons[neuron_id].get_output()

        return outputs

    def _process_layer(self, layer: List[str]) -> None:
        for neuron_id in layer:
            self.neurons[neuron_id].update()

    def get_network_state(self) -> Dict[str, Dict]:
        neuron_states = {neuron_id: neuron.get_state() for neuron_id, neuron in self.neurons.items()}
        synapse_states = [synapse.get_state() for synapse in self.synapses]

        return {
            "neurons": neuron_states,
            "synapses": synapse_states,
            "input_layer": self.input_layer,
            "hidden_layers": self.hidden_layers,
            "output_layer": self.output_layer
        }

    def save(self, filepath: str) -> None:
        state = self.get_network_state()
        np.save(filepath, state, allow_pickle=True)

    @classmethod
    def load(cls, filepath: str) -> 'Network':
        state = np.load(filepath, allow_pickle=True).item()

        network = cls(name=filepath.split('/')[-1].split('.')[0])

        return network


class BiologicalNeuralNetwork:
    """A biological neural network that simulates detailed neuron dynamics.
    This class implements a biologically detailed neural network with realistic
    neuron models, synapses, and 3D spatial organization."""

    def __init__(self):
        self.neurons = []  # List of neurons
        self.synapses = []  # List of synapses
        self.current_inputs = {}  # Dictionary of current inputs to neurons
        self.time = 0.0  # Current simulation time
        self.spike_history = {}  # Dictionary to store spike times for each neuron
        self.membrane_history = {}  # Dictionary to store membrane potential history
        self.groups = {}  # Dictionary to store neuron groups

    def add_neuron(self, neuron, group=None):
        """Add a neuron to the network.

        Args:
            neuron: A neuron object (e.g., HodgkinHuxleyNeuron)
            group: Optional group name for the neuron

        Returns:
            Index of the added neuron
        """
        # Initialize histories for the neuron
        neuron_idx = len(self.neurons)
        self.spike_history[neuron_idx] = []
        self.membrane_history[neuron_idx] = []

        # Add to group if specified
        if group is not None:
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(neuron_idx)

        self.neurons.append(neuron)
        return neuron_idx

    def connect(self, source_idx, target_idx, synapse_class, **synapse_params):
        """Connect two neurons with a synapse.

        Args:
            source_idx: Index of the source neuron
            target_idx: Index of the target neuron
            synapse_class: Class of synapse to create
            **synapse_params: Parameters for the synapse

        Returns:
            Index of the created synapse
        """
        # Create the synapse
        synapse = synapse_class(
            pre_neuron=self.neurons[source_idx],
            post_neuron=self.neurons[target_idx],
            **synapse_params
        )

        # Add the synapse to the network
        synapse_idx = len(self.synapses)
        self.synapses.append(synapse)

        # Register the synapse with the target neuron
        # Check if the neuron has the add_incoming_synapse or add_synapse method
        if hasattr(self.neurons[target_idx], 'add_incoming_synapse'):
            self.neurons[target_idx].add_incoming_synapse(synapse)
        elif hasattr(self.neurons[target_idx], 'add_synapse'):
            self.neurons[target_idx].add_synapse(synapse)

        return synapse_idx

    def inject_current(self, neuron_idx, current_func):
        """Inject current into a neuron.

        Args:
            neuron_idx: Index of the neuron
            current_func: Function that takes time and returns current value
        """
        self.current_inputs[neuron_idx] = current_func

    def step(self, dt):
        """Advance the simulation by one timestep.

        Args:
            dt: Time step in milliseconds
        """
        # Apply current inputs
        for idx, current_func in self.current_inputs.items():
            current = current_func(self.time)
            self.neurons[idx].inject_current(current)

        # Update all neurons
        for i, neuron in enumerate(self.neurons):
            # Update the neuron
            neuron.update(dt)

            # Record membrane potential
            self.membrane_history[i].append((self.time, neuron.v))

            # Check for spikes
            if neuron.has_spiked():
                self.spike_history[i].append(self.time)

        # Update all synapses
        for synapse in self.synapses:
            synapse.update(dt)

        # Advance simulation time
        self.time += dt

    def run(self, duration, dt=0.1, callback=None):
        """Run the simulation for a specified duration.

        Args:
            duration: Duration in milliseconds
            dt: Time step in milliseconds
            callback: Optional callback function called after each step
        """
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt)
            if callback is not None:
                callback(self)

    def get_spike_counts(self, start_time=None, end_time=None):
        """Get the number of spikes for each neuron in a time window.

        Args:
            start_time: Optional start time for counting
            end_time: Optional end time for counting

        Returns:
            Dictionary mapping neuron indices to spike counts
        """
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = self.time

        counts = {}
        for idx, spikes in self.spike_history.items():
            counts[idx] = sum(1 for t in spikes if start_time <= t <= end_time)

        return counts

    def get_firing_rates(self, window=100.0):
        """Calculate firing rates for all neurons.

        Args:
            window: Time window in milliseconds for rate calculation

        Returns:
            Dictionary mapping neuron indices to firing rates (Hz)
        """
        end_time = self.time
        start_time = max(0, end_time - window)
        duration_sec = (end_time - start_time) / 1000.0  # Convert to seconds

        rates = {}
        spike_counts = self.get_spike_counts(start_time, end_time)

        for idx, count in spike_counts.items():
            rates[idx] = count / duration_sec if duration_sec > 0 else 0

        return rates

    def create_small_world_network(self, n_neurons, k, p_rewire, neuron_class, synapse_class, **params):
        """Create a small-world network topology.

        Args:
            n_neurons: Number of neurons
            k: Each node is connected to k nearest neighbors
            p_rewire: Probability of rewiring each edge
            neuron_class: Class to use for neurons
            synapse_class: Class to use for synapses
            **params: Additional parameters for neurons/synapses

        Returns:
            Indices of created neurons
        """
        # Create a small-world graph
        graph = nx.watts_strogatz_graph(n=n_neurons, k=k, p=p_rewire)

        # Extract neuron and synapse parameters
        neuron_params = {k: v for k, v in params.items() 
                        if k not in ['weight', 'reversal_potential']}
        synapse_params = {k: v for k, v in params.items() 
                         if k in ['weight', 'reversal_potential']}

        # Add neurons
        neuron_indices = []
        for i in range(n_neurons):
            # Random position in 3D space
            position = (random.uniform(-100, 100), 
                       random.uniform(-100, 100), 
                       random.uniform(-100, 100))

            # Create neuron
            neuron = neuron_class(position=position, neuron_id=i, **neuron_params)
            idx = self.add_neuron(neuron)
            neuron_indices.append(idx)

        # Add synapses according to the graph
        for u, v in graph.edges():
            # Default synapse parameters if not specified
            params = dict(synapse_params)
            if 'weight' not in params:
                params['weight'] = random.uniform(0.1, 0.5)
            if 'reversal_potential' not in params:
                # 80% excitatory, 20% inhibitory
                params['reversal_potential'] = 0.0 if random.random() < 0.8 else -80.0

            self.connect(u, v, synapse_class, **params)

        return neuron_indices
