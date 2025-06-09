"""Plotting tools for neural data visualization.

This module provides various plots for analyzing neural network activity,
including raster plots, heat maps, and connectivity diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx


def plot_membrane_potentials(neurons, time_range=None, figsize=(10, 6)):
    """Plot membrane potentials of multiple neurons over time.

    Args:
        neurons: List of neuron objects
        time_range: Optional (start_time, end_time) tuple to limit the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, neuron in enumerate(neurons):
        times = np.array(neuron.t_history)
        voltages = np.array(neuron.v_history)

        # Filter by time range if specified
        if time_range is not None:
            start, end = time_range
            mask = (times >= start) & (times <= end)
            times = times[mask]
            voltages = voltages[mask]

        # Plot with a different color for each neuron
        ax.plot(times, voltages, label=f"Neuron {neuron.neuron_id}", 
               alpha=0.8, linewidth=1.5)

    # Add spike markers
    for i, neuron in enumerate(neurons):
        spike_times = neuron.spike_times
        if time_range is not None:
            start, end = time_range
            spike_times = [t for t in spike_times if start <= t <= end]

        if spike_times:
            # Plot small markers at spike times at a fixed voltage (30mV)
            ax.plot(spike_times, [30] * len(spike_times), 'o', 
                   markersize=4, color=f'C{i}', alpha=0.8)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Neuron Membrane Potentials')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    # Set y-axis limits to include typical range of membrane potentials
    ax.set_ylim(-90, 40)

    # Add threshold line
    if neurons:
        threshold = neurons[0].spike_threshold
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'Spike Threshold ({threshold} mV)')

    plt.tight_layout()
    return fig


def create_spike_raster(neurons, time_range=None, figsize=(10, 6)):
    """Create a raster plot of neuron spikes.

    Args:
        neurons: List of neuron objects
        time_range: Optional (start_time, end_time) tuple to limit the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Collect spike data
    spike_data = []
    neuron_ids = []

    for i, neuron in enumerate(neurons):
        spike_times = neuron.spike_times
        if time_range is not None:
            start, end = time_range
            spike_times = [t for t in spike_times if start <= t <= end]

        # Add to data
        spike_data.append(spike_times)
        neuron_ids.append(neuron.neuron_id)

    # Plot raster
    for i, (neuron_id, spike_times) in enumerate(zip(neuron_ids, spike_data)):
        if spike_times:  # Only plot if there are spikes
            ax.eventplot(spike_times, lineoffsets=i+1, linelengths=0.5, 
                        linewidths=1.5, colors=f'C{i%10}')

    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster Plot')

    # Set y-ticks to show neuron IDs
    ax.set_yticks(range(1, len(neurons)+1))
    ax.set_yticklabels(neuron_ids)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_connectivity_matrix(network, figsize=(8, 8)):
    """Plot the connectivity matrix of the network.

    Args:
        network: BiologicalNeuralNetwork object
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get connectivity matrix
    conn_matrix = network.get_connectivity_matrix()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create a custom colormap: blue for negative (inhibitory), 
    # red for positive (excitatory) weights
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    n_bins = 100
    cmap_name = 'exc_inh_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Plot matrix
    im = ax.imshow(conn_matrix, cmap=cm, interpolation='nearest', 
                  aspect='equal', vmin=-abs(conn_matrix).max(), 
                  vmax=abs(conn_matrix).max())

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Synaptic Weight')

    # Set labels and title
    ax.set_xlabel('Postsynaptic Neuron ID')
    ax.set_ylabel('Presynaptic Neuron ID')
    ax.set_title('Network Connectivity Matrix')

    # Set ticks
    tick_step = max(1, len(network.neurons) // 10)  # Show at most ~10 ticks
    ax.set_xticks(range(0, len(network.neurons), tick_step))
    ax.set_yticks(range(0, len(network.neurons), tick_step))

    plt.tight_layout()
    return fig


def plot_network_graph(network, layout='spring', figsize=(10, 10)):
    """Plot the network as a graph.

    Args:
        network: BiologicalNeuralNetwork object
        layout: Graph layout algorithm ('spring', '3d', 'circular', etc.)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (neurons)
    for i, neuron in enumerate(network.neurons):
        G.add_node(i, position=neuron.position, energy=neuron.energy, 
                 alive=neuron.alive, v=neuron.v)

    # Add edges (synapses)
    for synapse in network.synapses:
        pre_id = synapse.pre_neuron.neuron_id
        post_id = synapse.post_neuron.neuron_id
        weight = synapse.weight

        # Determine if excitatory or inhibitory
        if hasattr(synapse, 'reversal_potential'):
            syn_type = 'excitatory' if synapse.reversal_potential > -40 else 'inhibitory'
        else:
            syn_type = 'unknown'

        G.add_edge(pre_id, post_id, weight=weight, type=syn_type)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node positions
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == '3d':
        # Use actual 3D positions of neurons
        pos = {i: (n.position[0], n.position[1]) for i, n in enumerate(network.neurons)}
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors based on membrane potential
    v_min, v_max = -80, 40
    node_colors = []
    for i in G.nodes():
        v = network.neurons[i].v
        # Normalize to 0-1 range
        v_norm = (v - v_min) / (v_max - v_min)
        v_norm = np.clip(v_norm, 0, 1)

        # Blue (-80mV) to Red (+40mV)
        node_colors.append((1-v_norm, 0, v_norm))

    # Node sizes based on energy
    node_sizes = [network.neurons[i].energy * 5 + 50 for i in G.nodes()]

    # Edge colors based on synapse type
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['type'] == 'excitatory':
            edge_colors.append('green')
        elif data['type'] == 'inhibitory':
            edge_colors.append('red')
        else:
            edge_colors.append('gray')

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                         node_size=node_sizes, alpha=0.8)

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, 
                         edge_color=edge_colors, width=1.5, alpha=0.6)

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Remove axis
    ax.set_axis_off()

    # Title
    plt.title('Neural Network Graph')

    # Add legend for edge colors
    handles = [
        plt.Line2D([0], [0], color='green', lw=2, label='Excitatory'),
        plt.Line2D([0], [0], color='red', lw=2, label='Inhibitory'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Unknown')
    ]
    plt.legend(handles=handles)

    plt.tight_layout()
    return fig


def create_activity_animation(neurons, time_range, interval=100, figsize=(10, 6)):
    """Create an animation of neuron activity over time.

    Args:
        neurons: List of neuron objects
        time_range: (start_time, end_time) tuple
        interval: Animation interval in milliseconds
        figsize: Figure size

    Returns:
        Matplotlib animation
    """
    start_time, end_time = time_range

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize lines for each neuron
    lines = []
    for i, neuron in enumerate(neurons):
        line, = ax.plot([], [], label=f"Neuron {neuron.neuron_id}", 
                       alpha=0.8, linewidth=1.5, color=f'C{i%10}')
        lines.append(line)

    # Add spike markers (will be updated in animation)
    spike_markers = []
    for i, neuron in enumerate(neurons):
        marker, = ax.plot([], [], 'o', markersize=4, color=f'C{i%10}', alpha=0.8)
        spike_markers.append(marker)

    # Set up the plot
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Neuron Membrane Potentials')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_ylim(-90, 40)
    ax.set_xlim(start_time, start_time + 100)  # Initial window width: 100ms

    # Add threshold line
    if neurons:
        threshold = neurons[0].spike_threshold
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'Spike Threshold ({threshold} mV)')

    # Current time indicator
    time_line = ax.axvline(x=start_time, color='k', linestyle='-', alpha=0.5)
    time_text = ax.text(0.02, 0.98, f"Time: {start_time:.1f} ms", 
                       transform=ax.transAxes, verticalalignment='top')

    def init():
        """Initialize animation."""
        for line in lines:
            line.set_data([], [])
        for marker in spike_markers:
            marker.set_data([], [])
        time_line.set_xdata([start_time, start_time])
        time_text.set_text(f"Time: {start_time:.1f} ms")
        return lines + spike_markers + [time_line, time_text]

    def animate(frame):
        """Update animation for each frame."""
        current_time = start_time + frame * (end_time - start_time) / 100

        # Update time indicator
        time_line.set_xdata([current_time, current_time])
        time_text.set_text(f"Time: {current_time:.1f} ms")

        # Update x-axis window to follow current time
        window_width = 100  # ms
        ax.set_xlim(max(start_time, current_time - window_width/2), 
                   min(end_time, current_time + window_width/2))

        # Update neuron lines and spike markers
        for i, neuron in enumerate(neurons):
            # Filter times up to current time
            times = np.array(neuron.t_history)
            voltages = np.array(neuron.v_history)

            mask = times <= current_time
            plot_times = times[mask]
            plot_voltages = voltages[mask]

            lines[i].set_data(plot_times, plot_voltages)

            # Update spike markers
            spike_times = [t for t in neuron.spike_times if t <= current_time]
            if spike_times:
                spike_markers[i].set_data(spike_times, [30] * len(spike_times))
            else:
                spike_markers[i].set_data([], [])

        return lines + spike_markers + [time_line, time_text]

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=100, init_func=init, 
                                blit=True, interval=interval, repeat=True)

    plt.tight_layout()
    return ani


def plot_synapse_weights(synapses, time_range=None, figsize=(10, 6)):
    """Plot the weight changes of synapses over time (for STDP synapses).

    Args:
        synapses: List of synapse objects
        time_range: Optional (start_time, end_time) tuple to limit the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Filter synapses to only include those with weight history
    stdp_synapses = [s for s in synapses if hasattr(s, 'weight_history') and s.weight_history]

    if not stdp_synapses:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for i, synapse in enumerate(stdp_synapses):
        # Extract time and weight from history
        times, weights = zip(*synapse.weight_history)

        # Filter by time range if specified
        if time_range is not None:
            start, end = time_range
            time_weights = [(t, w) for t, w in zip(times, weights) if start <= t <= end]
            if time_weights:  # Only proceed if there are points within range
                times, weights = zip(*time_weights)
            else:
                continue

        # Plot with a different color for each synapse
        label = f"Syn: {synapse.pre_neuron.neuron_id} → {synapse.post_neuron.neuron_id}"
        ax.plot(times, weights, label=label, alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Synaptic Weight')
    ax.set_title('STDP Synaptic Weight Changes')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def create_firing_rate_heatmap(neurons, time_range, bin_width=10, figsize=(12, 8)):
    """Create a heatmap of neuron firing rates over time.

    Args:
        neurons: List of neuron objects
        time_range: (start_time, end_time) tuple
        bin_width: Width of time bins in ms
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    start_time, end_time = time_range

    # Create time bins
    bins = np.arange(start_time, end_time + bin_width, bin_width)
    n_bins = len(bins) - 1

    # Create matrix to hold spike counts
    n_neurons = len(neurons)
    spike_counts = np.zeros((n_neurons, n_bins))

    # Count spikes in each bin for each neuron
    for i, neuron in enumerate(neurons):
        spike_times = [t for t in neuron.spike_times if start_time <= t <= end_time]
        if spike_times:
            hist, _ = np.histogram(spike_times, bins=bins)
            spike_counts[i, :] = hist

    # Convert to firing rates (spikes per second)
    firing_rates = spike_counts * (1000 / bin_width)  # Convert to Hz

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(firing_rates, aspect='auto', cmap='inferno', interpolation='nearest',
                  extent=[start_time, end_time, n_neurons-0.5, -0.5])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Firing Rate (Hz)')

    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Neuron Firing Rate Heatmap')

    # Set y-ticks to show neuron IDs
    neuron_ids = [n.neuron_id for n in neurons]
    ax.set_yticks(range(n_neurons))
    ax.set_yticklabels(neuron_ids)

    # Set x-ticks at reasonable intervals
    tick_interval = max(1, (end_time - start_time) // 10)  # At most 10 ticks
    ax.set_xticks(np.arange(start_time, end_time + 1, tick_interval))

    plt.tight_layout()
    return fig


def plot_ion_channel_dynamics(neuron, time_range=None, figsize=(10, 8)):
    """Plot the dynamics of ion channel gating variables.

    Args:
        neuron: HodgkinHuxleyNeuron object
        time_range: Optional (start_time, end_time) tuple to limit the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract data from neuron history
    if not hasattr(neuron, 't_history') or not neuron.t_history:
        return None

    # Filter by time range if specified
    times = np.array(neuron.t_history)
    if time_range is not None:
        start, end = time_range
        mask = (times >= start) & (times <= end)
        times = times[mask]

        # If no data in range, return None
        if len(times) == 0:
            return None

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Plot membrane potential
    axs[0].plot(times, np.array(neuron.v_history)[mask] if time_range else neuron.v_history, 
               'k-', label='V')
    axs[0].set_ylabel('Membrane\nPotential (mV)')
    axs[0].set_title(f'Hodgkin-Huxley Dynamics for Neuron {neuron.neuron_id}')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_ylim(-90, 40)

    # Plot gating variables if available
    if hasattr(neuron, 'm_history') and neuron.m_history:
        m_values = np.array(neuron.m_history)[mask] if time_range else neuron.m_history
        h_values = np.array(neuron.h_history)[mask] if time_range else neuron.h_history
        n_values = np.array(neuron.n_history)[mask] if time_range else neuron.n_history

        # Plot sodium activation (m)
        axs[1].plot(times, m_values, 'b-', label='m')
        axs[1].set_ylabel('Na+ Activation\nm')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_ylim(0, 1)

        # Plot sodium inactivation (h)
        axs[2].plot(times, h_values, 'r-', label='h')
        axs[2].set_ylabel('Na+ Inactivation\nh')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].set_ylim(0, 1)

        # Plot potassium activation (n)
        axs[3].plot(times, n_values, 'g-', label='n')
        axs[3].set_ylabel('K+ Activation\nn')
        axs[3].set_xlabel('Time (ms)')
        axs[3].grid(True, linestyle='--', alpha=0.7)
        axs[3].set_ylim(0, 1)

        # Add legend to each subplot
        for ax in axs:
            ax.legend(loc='upper right')
    else:
        # If gating variables not available, show a message
        for i in range(1, 4):
            axs[i].text(0.5, 0.5, 'Gating variable data not available', 
                       ha='center', va='center', transform=axs[i].transAxes)
            axs[i].set_yticks([])

    plt.tight_layout()
    return fig
