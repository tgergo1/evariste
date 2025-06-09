#!/usr/bin/env python3
"""
Hodgkin-Huxley model demonstration.

This script demonstrates the behavior of a single Hodgkin-Huxley neuron
under different stimulation conditions.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Add the parent directory to the path to import evariste modules
sys.path.append('..')
from evariste.models.hodgkin_huxley import HodgkinHuxleyNeuron


def single_neuron_simulation():
    """Simulate a single Hodgkin-Huxley neuron with varying current."""
    # Create neuron
    neuron = HodgkinHuxleyNeuron(position=(0, 0, 0), neuron_id=0)

    # Simulation parameters
    duration = 100.0  # ms
    dt = 0.01  # ms

    # Record time and voltage
    times = np.arange(0, duration, dt)
    voltages = []
    m_values = []
    h_values = []
    n_values = []

    # Stimulation protocol: vary current over time
    def current_func(t):
        if 10 <= t < 20:  # First pulse
            return 10.0
        elif 40 <= t < 50:  # Second pulse (stronger)
            return 20.0
        elif 70 <= t < 80:  # Third pulse (negative)
            return -5.0
        else:
            return 0.0

    # Set up the neuron with this current function
    neuron.inject_current(current_func)

    # Add history tracking
    neuron.m_history = []
    neuron.h_history = []
    neuron.n_history = []

    # Run simulation
    print("Running Hodgkin-Huxley simulation...")
    for t in times:
        # Update neuron state
        neuron.update(dt, t)

        # Record values
        voltages.append(neuron.v)
        m_values.append(neuron.m)
        h_values.append(neuron.h)
        n_values.append(neuron.n)

        # Store in neuron's history
        neuron.m_history.append(neuron.m)
        neuron.h_history.append(neuron.h)
        neuron.n_history.append(neuron.n)

    # Calculate currents
    na_current = [neuron.i_na(v, m, h) for v, m, h in zip(voltages, m_values, h_values)]
    k_current = [neuron.i_k(v, n) for v, n in zip(voltages, n_values)]
    leak_current = [neuron.i_l(v) for v in voltages]

    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Plot membrane potential
    axs[0].plot(times, voltages, 'k-')
    axs[0].set_ylabel('Membrane\nPotential (mV)')
    axs[0].set_title('Hodgkin-Huxley Model Simulation')
    axs[0].grid(True)

    # Plot gating variables
    axs[1].plot(times, m_values, 'b-', label='m (Na+ activation)')
    axs[1].plot(times, h_values, 'r-', label='h (Na+ inactivation)')
    axs[1].plot(times, n_values, 'g-', label='n (K+ activation)')
    axs[1].set_ylabel('Gating\nVariables')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Plot injected current
    current_values = [current_func(t) for t in times]
    axs[2].plot(times, current_values, 'k-')
    axs[2].set_ylabel('Injected\nCurrent (nA)')
    axs[2].grid(True)

    # Plot ion channel currents
    axs[3].plot(times, na_current, 'b-', label='Na+ current')
    axs[3].plot(times, k_current, 'g-', label='K+ current')
    axs[3].plot(times, leak_current, 'r-', label='Leak current')
    axs[3].set_ylabel('Channel\nCurrents (nA)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True)

    # Plot total current
    total_current = [i + k + l + c for i, k, l, c in 
                   zip(na_current, k_current, leak_current, current_values)]
    axs[4].plot(times, total_current, 'k-')
    axs[4].set_ylabel('Total\nCurrent (nA)')
    axs[4].set_xlabel('Time (ms)')
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

    return neuron, times, voltages


def animated_simulation():
    """Create an animated visualization of the HH model."""
    # Create neuron
    neuron = HodgkinHuxleyNeuron(position=(0, 0, 0), neuron_id=0)

    # Simulation parameters
    duration = 100.0  # ms
    dt = 0.01  # ms

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Hodgkin-Huxley Neuron Animation', fontsize=16)

    # Voltage plot
    line_v, = ax1.plot([], [], 'k-', lw=2)
    ax1.set_xlim(0, duration)
    ax1.set_ylim(-90, 40)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.grid(True)

    # Gate variables plot
    line_m, = ax2.plot([], [], 'b-', lw=2, label='m (Na+ activation)')
    line_h, = ax2.plot([], [], 'r-', lw=2, label='h (Na+ inactivation)')
    line_n, = ax2.plot([], [], 'g-', lw=2, label='n (K+ activation)')
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Gating Variables')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Stimulus current function
    def current_func(t):
        if 10 <= t < 15:  # First pulse
            return 10.0
        elif 30 <= t < 35:  # Second pulse
            return 15.0
        elif 50 <= t < 55:  # Third pulse
            return 20.0
        elif 70 <= t < 75:  # Fourth pulse (negative)
            return -5.0
        else:
            return 0.0

    # Set up the neuron with this current function
    neuron.inject_current(current_func)

    # Initialize data arrays
    times = []
    voltages = []
    m_values = []
    h_values = []
    n_values = []

    def init():
        """Initialize the animation."""
        line_v.set_data([], [])
        line_m.set_data([], [])
        line_h.set_data([], [])
        line_n.set_data([], [])
        return line_v, line_m, line_h, line_n

    def animate(frame):
        """Update the animation for each frame."""
        global neuron, times, voltages, m_values, h_values, n_values

        # Current time
        t = frame * dt
        times.append(t)

        # Update neuron
        neuron.update(dt, t)

        # Record values
        voltages.append(neuron.v)
        m_values.append(neuron.m)
        h_values.append(neuron.h)
        n_values.append(neuron.n)

        # Update plots
        line_v.set_data(times, voltages)
        line_m.set_data(times, m_values)
        line_h.set_data(times, h_values)
        line_n.set_data(times, n_values)

        return line_v, line_m, line_h, line_n

    # Create animation
    frames = int(duration / dt)
    ani = FuncAnimation(fig, animate, frames=frames, init_func=init, 
                      interval=20, blit=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for title
    plt.show()


def frequency_response_test():
    """Test neuron response to different current amplitudes."""
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Current amplitudes to test
    current_amplitudes = [5.0, 10.0, 15.0]
    colors = ['blue', 'green', 'red']

    # Run simulation for each current amplitude
    for i, amp in enumerate(current_amplitudes):
        # Create neuron
        neuron = HodgkinHuxleyNeuron(position=(0, 0, 0), neuron_id=i)

        # Simulation parameters
        duration = 100.0  # ms
        dt = 0.01  # ms

        # Constant current injection
        neuron.inject_current(amp)

        # Record time and voltage
        times = np.arange(0, duration, dt)
        voltages = []

        # Run simulation
        for t in times:
            neuron.update(dt, t)
            voltages.append(neuron.v)

        # Plot voltage trace
        axs[0].plot(times, voltages, color=colors[i], 
                   label=f'I = {amp} nA')

        # Count spikes
        spike_times = neuron.spike_times
        axs[1].plot(spike_times, [amp] * len(spike_times), 'o', 
                   color=colors[i], markersize=8)

        # Calculate firing rate (if there are spikes)
        if len(spike_times) > 1:
            intervals = np.diff(spike_times)
            mean_interval = np.mean(intervals)
            firing_rate = 1000 / mean_interval  # Convert to Hz
        else:
            firing_rate = 0

        print(f"Current: {amp} nA, Spikes: {len(spike_times)}, "  
              f"Rate: {firing_rate:.2f} Hz")

    # Customize plots
    axs[0].set_ylabel('Membrane\nPotential (mV)')
    axs[0].set_title('Neuron Response to Different Current Amplitudes')
    axs[0].grid(True)
    axs[0].legend(loc='upper right')

    axs[1].set_ylabel('Current\nAmplitude (nA)')
    axs[1].set_yticks(current_amplitudes)
    axs[1].set_title('Spike Raster')
    axs[1].grid(True)

    # Create F-I curve with more points
    more_amplitudes = np.linspace(0, 20, 21)
    firing_rates = []

    for amp in more_amplitudes:
        neuron = HodgkinHuxleyNeuron(position=(0, 0, 0), neuron_id=0)
        neuron.inject_current(amp)

        # Shorter simulation for efficiency
        duration = 200.0  # ms
        dt = 0.01  # ms

        # Run simulation
        for t in np.arange(0, duration, dt):
            neuron.update(dt, t)

        # Calculate firing rate (after initial transient)
        spike_times = [t for t in neuron.spike_times if t > 50]
        if len(spike_times) > 1:
            intervals = np.diff(spike_times)
            mean_interval = np.mean(intervals)
            rate = 1000 / mean_interval  # Convert to Hz
        else:
            rate = 0

        firing_rates.append(rate)

    # Plot F-I curve
    axs[2].plot(more_amplitudes, firing_rates, 'ko-')
    axs[2].set_xlabel('Current Amplitude (nA)')
    axs[2].set_ylabel('Firing Rate (Hz)')
    axs[2].set_title('F-I Curve')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Hodgkin-Huxley Neuron Model Demonstration")
    print("====================================")
    print("1. Single neuron simulation")
    print("2. Animated simulation")
    print("3. Frequency response test")

    choice = input("\nEnter your choice (1-3): ")

    if choice == '1':
        single_neuron_simulation()
    elif choice == '2':
        animated_simulation()
    elif choice == '3':
        frequency_response_test()
    else:
        print("Invalid choice. Exiting.")
