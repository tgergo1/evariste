#!/usr/bin/env python3
"""
Biological neural network demonstration.

This script demonstrates the creation and simulation of a biologically
detailed neural network with 3D visualization.
"""

import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path to import evariste modules
sys.path.append('..')
from evariste.models.network import BiologicalNeuralNetwork
from evariste.models.hodgkin_huxley import HodgkinHuxleyNeuron
from evariste.models.synapses import ExponentialSynapse, STDPSynapse, CompoundSynapse
from evariste.visualization.neuron_3d import create_visualization_app
from evariste.visualization.plots import (
    plot_membrane_potentials, create_spike_raster,
    plot_connectivity_matrix, plot_network_graph,
    create_firing_rate_heatmap, plot_ion_channel_dynamics
)


def create_demo_network(network_type='random'):
    """Create a demonstration neural network.

    Args:
        network_type: Type of network to create ('random', 'small_world', 'layers')

    Returns:
        BiologicalNeuralNetwork object
    """
    network = BiologicalNeuralNetwork()

    if network_type == 'random':
        # Create a random network
        print("Creating random network...")
        n_neurons = 20

        # Create neurons with random parameters
        for i in range(n_neurons):
            # Random position in 3D space
            position = (random.uniform(-100, 100), 
                       random.uniform(-100, 100), 
                       random.uniform(-100, 100))

            # Random parameters
            params = {
                'g_na': random.uniform(100, 140),      # Sodium conductance
                'g_k': random.uniform(30, 42),         # Potassium conductance
                'g_l': random.uniform(0.2, 0.4),       # Leak conductance
                'c_m': random.uniform(0.8, 1.2)        # Membrane capacitance
            }

            # Create neuron
            neuron = HodgkinHuxleyNeuron(position=position, neuron_id=i, **params)
            network.add_neuron(neuron)

        # Connect neurons randomly with 10% probability
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and random.random() < 0.1:  # 10% connection probability
                    # Determine if excitatory or inhibitory
                    if random.random() < 0.8:  # 80% excitatory
                        syn_class = ExponentialSynapse
                        weight = random.uniform(0.1, 0.5)
                        reversal_potential = 0.0  # Excitatory
                    else:  # 20% inhibitory
                        syn_class = ExponentialSynapse
                        weight = random.uniform(0.1, 0.8)
                        reversal_potential = -80.0  # Inhibitory

                    # Create synapse
                    network.connect(i, j, synapse_class=syn_class, 
                                  weight=weight, reversal_potential=reversal_potential)

    elif network_type == 'small_world':
        # Create a small-world network using the built-in function
        print("Creating small-world network...")
        network.create_small_world_network(
            n_neurons=30,
            k=4,               # Mean degree (connections per neuron)
            p_rewire=0.1,      # Rewiring probability
            neuron_class=HodgkinHuxleyNeuron,
            synapse_class=ExponentialSynapse
        )

    elif network_type == 'layers':
        # Create a layered network (e.g., for feedforward processing)
        print("Creating layered network...")

        # Create three layers
        layer_sizes = [5, 8, 3]  # Input, hidden, output
        layer_positions = [-100, 0, 100]  # x-coordinates for layers

        # Create neurons in each layer
        layer_indices = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_neurons = []
            for i in range(size):
                # Position neurons in a grid for each layer
                x = layer_positions[layer_idx]
                spacing = 50
                y = (i - size/2) * spacing
                z = random.uniform(-20, 20)

                position = (x, y, z)
                neuron = HodgkinHuxleyNeuron(position=position, 
                                            neuron_id=len(network.neurons))
                idx = network.add_neuron(neuron, f"layer_{layer_idx}")
                layer_neurons.append(idx)

            layer_indices.append(layer_neurons)

        # Connect layers fully (each neuron in layer i connects to all in layer i+1)
        for i in range(len(layer_sizes) - 1):
            pre_layer = layer_indices[i]
            post_layer = layer_indices[i+1]

            for pre_idx in pre_layer:
                for post_idx in post_layer:
                    # All connections are excitatory with random weights
                    weight = random.uniform(0.2, 0.6)

                    # Use STDP for learning
                    network.connect(pre_idx, post_idx, synapse_class=STDPSynapse,
                                  weight=weight, reversal_potential=0.0)

    print(f"Created network with {len(network.neurons)} neurons and "  
          f"{len(network.synapses)} synapses")
    return network


def run_simulation(network, duration=500.0, dt=0.1, visualize_3d=True):
    """Run a simulation of the network.

    Args:
        network: BiologicalNeuralNetwork object
        duration: Simulation duration (ms)
        dt: Time step (ms)
        visualize_3d: Whether to show 3D visualization

    Returns:
        QApplication instance if 3D visualization is enabled
    """
    # Start time measurement
    start_time = time.time()

    # Initialize 3D visualization if requested
    if visualize_3d:
        app, visualizer = create_visualization_app(network)
    else:
        app = None

    # Set up input stimulus
    def input_stimulus(t):
        """Generate input current pulses."""
        # Regular pulses every 100ms
        if t % 100 < 5:  # 5ms pulse width
            return 10.0  # Strong pulse
        else:
            return 0.0

    # Inject current to a subset of neurons
    input_neurons = random.sample(range(len(network.neurons)), 
                                 max(1, len(network.neurons) // 5))  # ~20% of neurons

    for idx in input_neurons:
        network.inject_current(idx, input_stimulus)

    print(f"\nRunning simulation for {duration} ms...")

    # Run simulation with progress reporting
    progress_step = 10  # Report progress every 10% 
    next_report = progress_step

    for t in np.arange(0, duration, dt):
        # Update simulation
        network.step(dt)

        # Process 3D visualization events
        if visualize_3d and app is not None:
            app.processEvents()

        # Report progress
        progress = int(t / duration * 100)
        if progress >= next_report:
            elapsed = time.time() - start_time
            est_total = elapsed / (progress / 100)
            est_remaining = est_total - elapsed

            print(f"{progress}% complete ({t:.1f}/{duration} ms) - "  
                  f"Elapsed: {elapsed:.1f}s, Remaining: {est_remaining:.1f}s")
            next_report += progress_step

    # Report completion
    total_time = time.time() - start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds")

    # Count spikes
    spike_counts = network.get_spike_counts()
    total_spikes = sum(spike_counts.values())
    avg_rate = total_spikes / (len(network.neurons) * duration / 1000)  # Hz

    print(f"Total spikes: {total_spikes}")
    print(f"Average firing rate: {avg_rate:.2f} Hz")

    return app


def generate_analysis_plots(network, time_range=None):
    """Generate analysis plots for the network simulation.

    Args:
        network: BiologicalNeuralNetwork object
        time_range: Optional (start_time, end_time) tuple
    """
    if time_range is None:
        # Use the whole simulation time
        time_range = (0, network.time)

    # Select a subset of neurons to analyze (max 10 for clarity)
    if len(network.neurons) <= 10:
        selected_neurons = network.neurons
    else:
        # Select neurons with most spikes
        spike_counts = network.get_spike_counts(time_range[0], time_range[1])
        top_neurons = sorted(spike_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        selected_indices = [idx for idx, _ in top_neurons]
        selected_neurons = [network.neurons[idx] for idx in selected_indices]

    # Create all analysis plots
    plots = [
        # 1. Membrane potentials
        plot_membrane_potentials(selected_neurons, time_range),

        # 2. Spike raster
        create_spike_raster(network.neurons, time_range),

        # 3. Network connectivity
        plot_connectivity_matrix(network),

        # 4. Network graph
        plot_network_graph(network),

        # 5. Firing rate heatmap
        create_firing_rate_heatmap(network.neurons, time_range, bin_width=20),

        # 6. Ion channel dynamics for one neuron
        plot_ion_channel_dynamics(selected_neurons[0] if selected_neurons else None, time_range)
    ]

    # Show all plots
    for plot in plots:
        if plot is not None:  # Some plots might return None if data is missing
            plt.figure(plot.number)
            plt.show()


def interactive_demo():
    """Run an interactive demonstration of the biological neural network."""
    print("\nBiological Neural Network Interactive Demo")
    print("===========================================\n")

    # Network type selection
    print("Network Types:")
    print("1. Random network")
    print("2. Small-world network")
    print("3. Layered network")

    while True:
        choice = input("\nSelect network type (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    network_types = {
        '1': 'random',
        '2': 'small_world',
        '3': 'layers'
    }

    # Create the selected network
    network = create_demo_network(network_types[choice])

    # Visualization choice
    while True:
        vis_choice = input("\nEnable 3D visualization? (y/n): ").lower()
        if vis_choice in ['y', 'n']:
            break
        print("Invalid choice. Please enter y or n.")

    visualize_3d = (vis_choice == 'y')

    # Simulation duration
    while True:
        try:
            duration = float(input("\nSimulation duration (ms, 100-1000): "))
            if 100 <= duration <= 1000:
                break
            print("Duration must be between 100 and 1000 ms.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Run the simulation
    app = run_simulation(network, duration=duration, visualize_3d=visualize_3d)

    # Analysis plots
    while True:
        plot_choice = input("\nGenerate analysis plots? (y/n): ").lower()
        if plot_choice in ['y', 'n']:
            break
        print("Invalid choice. Please enter y or n.")

    if plot_choice == 'y':
        generate_analysis_plots(network)

    # Keep the application running if 3D visualization is enabled
    if visualize_3d and app is not None:
        print("\n3D visualization is running. Close the window to exit.")
        sys.exit(app.exec_())


if __name__ == "__main__":
    interactive_demo()
