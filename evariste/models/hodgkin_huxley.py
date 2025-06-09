"""Hodgkin-Huxley neuron model.

This module implements the classic Hodgkin-Huxley model of neuronal dynamics,
which is one of the most biologically accurate models of neuron behavior.
"""

import numpy as np
from scipy.integrate import odeint


class HodgkinHuxleyNeuron:
    """Hodgkin-Huxley neuron model.

    Models the dynamics of membrane potential with sodium and potassium ion channels.
    Uses the standard Hodgkin-Huxley differential equations and parameters.
    """

    def __init__(self, position, neuron_id, g_na=120.0, g_k=36.0, g_l=0.3, 
                 e_na=50.0, e_k=-77.0, e_l=-54.387, c_m=1.0):
        """Initialize a Hodgkin-Huxley neuron.

        Args:
            position: 3D position (x, y, z) of the neuron in space
            neuron_id: Unique identifier for the neuron
            g_na: Maximum sodium conductance (mS/cm^2)
            g_k: Maximum potassium conductance (mS/cm^2)
            g_l: Leak conductance (mS/cm^2)
            e_na: Sodium reversal potential (mV)
            e_k: Potassium reversal potential (mV)
            e_l: Leak reversal potential (mV)
            c_m: Membrane capacitance (uF/cm^2)
        """
        self.position = position
        self.neuron_id = neuron_id

        # Conductances (mS/cm^2)
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l

        # Reversal potentials (mV)
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l

        # Membrane capacitance (uF/cm^2)
        self.c_m = c_m

        # State variables
        self.v = -65.0  # Initial membrane potential (mV)
        self.m = 0.05   # Initial sodium activation gating variable
        self.h = 0.6    # Initial sodium inactivation gating variable
        self.n = 0.32   # Initial potassium activation gating variable

        # Record history
        self.v_history = [self.v]
        self.t_history = [0.0]

        # Injected current
        self.i_inj = 0.0

        # Synaptic inputs
        self.synaptic_inputs = []

        # Spike detection
        self.spike_threshold = 0.0  # mV
        self.last_spike_time = -1000.0  # Large negative number (ms)
        self.spike_times = []
        self.refractory_period = 2.0  # ms
        self.in_refractory = False

        # Morphology (for visualization)
        self.soma_radius = 10.0  # μm
        self.dendrites = []  # List of dendrite segments
        self.axon = []       # List of axon segments

        # Energy/metabolism
        self.energy = 100.0  # Arbitrary units
        self.energy_decay_rate = 0.01  # Energy consumed per ms
        self.spike_energy_cost = 1.0   # Energy consumed per spike

        # Living state
        self.alive = True

    def alpha_m(self, v):
        """Sodium activation rate."""
        return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0)) if v != -40.0 else 1.0

    def beta_m(self, v):
        """Sodium deactivation rate."""
        return 4.0 * np.exp(-(v + 65.0) / 18.0)

    def alpha_h(self, v):
        """Sodium inactivation rate."""
        return 0.07 * np.exp(-(v + 65.0) / 20.0)

    def beta_h(self, v):
        """Sodium deinactivation rate."""
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

    def alpha_n(self, v):
        """Potassium activation rate."""
        return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0)) if v != -55.0 else 0.1

    def beta_n(self, v):
        """Potassium deactivation rate."""
        return 0.125 * np.exp(-(v + 65.0) / 80.0)

    def i_na(self, v, m, h):
        """Sodium current."""
        return self.g_na * m**3 * h * (v - self.e_na)

    def i_k(self, v, n):
        """Potassium current."""
        return self.g_k * n**4 * (v - self.e_k)

    def i_l(self, v):
        """Leak current."""
        return self.g_l * (v - self.e_l)

    def i_syn(self, t):
        """Total synaptic current at time t."""
        i_total = 0.0
        for syn in self.synaptic_inputs:
            i_total += syn.current(t, self.v)
        return i_total

    def dxdt(self, x, t):
        """Compute derivatives for the Hodgkin-Huxley model.

        Args:
            x: State vector [v, m, h, n]
            t: Time (ms)

        Returns:
            Derivatives [dv/dt, dm/dt, dh/dt, dn/dt]
        """
        v, m, h, n = x

        # Calculate membrane currents
        i_na_val = self.i_na(v, m, h)
        i_k_val = self.i_k(v, n)
        i_l_val = self.i_l(v)
        i_syn_val = self.i_syn(t)

        # External injected current (can be time-dependent)
        i_inj = self.i_inj if not callable(self.i_inj) else self.i_inj(t)

        # Calculate derivatives
        dv_dt = (i_inj - i_na_val - i_k_val - i_l_val + i_syn_val) / self.c_m
        dm_dt = self.alpha_m(v) * (1.0 - m) - self.beta_m(v) * m
        dh_dt = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn_dt = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n

        return [dv_dt, dm_dt, dh_dt, dn_dt]

    def update(self, dt=0.01, t_current=None):
        """Update neuron state for a time step.

        Args:
            dt: Time step (ms)
            t_current: Current simulation time (ms)

        Returns:
            Updated membrane potential
        """
        if not self.alive:
            return self.v

        # Update energy
        self.energy -= self.energy_decay_rate * dt
        if self.energy <= 0:
            self.die()
            return self.v

        # Check if in refractory period
        if t_current is not None and t_current - self.last_spike_time < self.refractory_period:
            self.in_refractory = True
            # During refractoriness, membrane potential is close to potassium reversal potential
            self.v = -65.0
            return self.v
        else:
            self.in_refractory = False

        # Current state
        x0 = [self.v, self.m, self.h, self.n]

        # Calculate next state
        t = np.array([0, dt])
        result = odeint(self.dxdt, x0, t)

        # Update state variables
        self.v, self.m, self.h, self.n = result[-1]

        # Record history
        if t_current is not None:
            self.v_history.append(self.v)
            self.t_history.append(t_current)

        # Check for spike
        if self.v > self.spike_threshold and not self.in_refractory:
            if t_current is not None:
                if t_current - self.last_spike_time > self.refractory_period:
                    self.spike_times.append(t_current)
                    self.last_spike_time = t_current
                    self.energy -= self.spike_energy_cost
                    # Trigger synaptic release
                    self.send_spike(t_current)

        return self.v

    def inject_current(self, current):
        """Inject current into the neuron.

        Args:
            current: Current value (nA) or a function of time
        """
        self.i_inj = current

    def add_synapse(self, synapse):
        """Add a synaptic input to this neuron.

        Args:
            synapse: Synapse object connecting to this neuron
        """
        self.synaptic_inputs.append(synapse)

    def send_spike(self, t):
        """Send spike to all outgoing synapses.

        Args:
            t: Current time (ms)
        """
        # This would be implemented by the Network class
        pass

    def add_dendrite(self, start_point, end_point, diameter=1.0):
        """Add a dendrite segment to the neuron.

        Args:
            start_point: (x, y, z) starting point
            end_point: (x, y, z) ending point
            diameter: Diameter of dendrite (μm)
        """
        self.dendrites.append({
            'start': start_point,
            'end': end_point,
            'diameter': diameter
        })

    def add_axon_segment(self, start_point, end_point, diameter=0.5):
        """Add an axon segment to the neuron.

        Args:
            start_point: (x, y, z) starting point
            end_point: (x, y, z) ending point
            diameter: Diameter of axon (μm)
        """
        self.axon.append({
            'start': start_point,
            'end': end_point,
            'diameter': diameter
        })

    def die(self):
        """Mark the neuron as dead."""
        print(f"Neuron {self.neuron_id} at position {self.position} has died.")
        self.alive = False

    def get_state(self):
        """Get the full state of the neuron.

        Returns:
            Dictionary with all state variables
        """
        return {
            'neuron_id': self.neuron_id,
            'position': self.position,
            'v': self.v,
            'm': self.m,
            'h': self.h,
            'n': self.n,
            'energy': self.energy,
            'alive': self.alive,
            'spike_times': self.spike_times,
            'in_refractory': self.in_refractory
        }
