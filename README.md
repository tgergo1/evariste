![alt text](https://raw.githubusercontent.com/tgergo1/evariste/master/misc/logo.png "evariste")
# Evariste: Biologically Accurate Neural Simulation

Evariste is a bold approach to general artificial intelligence through biologically accurate neural simulation. This project implements detailed neuron models and realistic neural networks with a focus on biological fidelity, not just computational efficiency.

## Overview

Unlike traditional artificial neural networks, Evariste models neurons at the cellular level, simulating actual biological processes including:

- **Hodgkin-Huxley neurons**: Complete with ion channels, membrane potentials, and action potentials
- **Realistic morphology**: Neurons with detailed dendrites, soma, and axons in 3D space
- **Biological synapses**: Including AMPA, NMDA, and GABA receptor dynamics
- **Synaptic plasticity**: STDP (Spike-Timing-Dependent Plasticity) for learning
- **Spatial connectivity**: Networks organized with realistic 3D topologies

## Components

### Models
Detailled biological models including:
- Hodgkin-Huxley neuron model
- Various synapse types (exponential, STDP, compound)
- Complete neural networks with different topologies

### Visualization
Comprehensive tools for visualizing neural activity:
- 3D real-time visualization of neurons and their morphology
- Membrane potential plots and raster diagrams
- Heat maps of neural activity
- Network connectivity visualization

### Examples
Ready-to-run demonstrations:
- Single neuron simulations
- Small and large network examples
- Various network topologies (random, small-world, layered)

## Getting Started

### Requirements
- Python 3.8 or higher
- Dependencies: numpy, scipy, matplotlib, PyQt5, pyqtgraph, PyOpenGL, networkx

### Installation

```bash
# Clone the repository
git clone https://github.com/tgergo1/evariste.git
cd evariste

# Install dependencies
pip install -e .

# For additional simulation capabilities
pip install -e .[neuron,brian2]
```

### Running Demos

```bash
# Run the Hodgkin-Huxley neuron demo
python -m evariste.examples.hodgkin_huxley_demo

# Run the neural network demo with 3D visualization
python -m evariste.examples.neural_network_demo
```

## Usage Examples

### Creating a Hodgkin-Huxley Neuron

```python
from evariste.models.hodgkin_huxley import HodgkinHuxleyNeuron

# Create a neuron with custom parameters
neuron = HodgkinHuxleyNeuron(
    position=(0, 0, 0),
    neuron_id=1,
    g_na=120.0,    # Sodium conductance
    g_k=36.0,      # Potassium conductance
    g_l=0.3,       # Leak conductance
    c_m=1.0        # Membrane capacitance
)

# Inject current and update
neuron.inject_current(10.0)  # 10 nA
neuron.update(dt=0.01)  # 0.01 ms time step
print(f"Membrane potential: {neuron.v} mV")
```

### Creating a Small Neural Network

```python
from evariste.models.network import BiologicalNeuralNetwork
from evariste.models.hodgkin_huxley import HodgkinHuxleyNeuron
from evariste.models.synapses import ExponentialSynapse

# Create a network
network = BiologicalNeuralNetwork()

# Add neurons
idx1 = network.add_neuron(HodgkinHuxleyNeuron(position=(0, 0, 0), neuron_id=0))
idx2 = network.add_neuron(HodgkinHuxleyNeuron(position=(50, 0, 0), neuron_id=1))

# Connect neurons with an excitatory synapse
network.connect(idx1, idx2, synapse_class=ExponentialSynapse, 
               weight=0.5, reversal_potential=0.0)

# Run simulation for 100 ms
network.run(duration=100.0, dt=0.1)
```

### Visualizing a Network

```python
from evariste.visualization.neuron_3d import create_visualization_app

# Create and show 3D visualization
app, visualizer = create_visualization_app(network)
app.exec_()
```

## Contribution Guidelines

This project adheres to Evariste's [code of conduct](code_of_conduct.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/tgergo1/evariste/issues) for tracking requests and bugs.

The Evariste project strives to abide by generally accepted best practices in open-source software development:

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code_of_conduct.md)

## License

See the [LICENSE](LICENSE) file for details.
