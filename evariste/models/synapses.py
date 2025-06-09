"""Synapse models for the Evariste system.

This module contains different types of biologically realistic synapses
including conductance-based and current-based models with plasticity.
"""

import numpy as np


class Synapse:
    """Base class for all synapses."""

    def __init__(self, pre_neuron, post_neuron, weight=0.5):
        """Initialize a synapse between two neurons.

        Args:
            pre_neuron: Presynaptic neuron (source)
            post_neuron: Postsynaptic neuron (target)
            weight: Synaptic weight (strength of connection)
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = 1.0  # ms
        self.last_spike_time = -1000.0  # Large negative number
        self.active = True

    def receive_spike(self, t):
        """Receive a spike from the presynaptic neuron.

        Args:
            t: Time of spike (ms)
        """
        self.last_spike_time = t

    def current(self, t, v_post):
        """Calculate the current flowing through the synapse.

        Args:
            t: Current time (ms)
            v_post: Postsynaptic membrane potential (mV)

        Returns:
            Synaptic current (nA)
        """
        raise NotImplementedError("Subclasses must implement current()")


class ExponentialSynapse(Synapse):
    """Synapse with exponential decay of current after a spike."""

    def __init__(self, pre_neuron, post_neuron, weight=0.5, tau=2.0, reversal_potential=0.0):
        """Initialize an exponential synapse.

        Args:
            pre_neuron: Presynaptic neuron
            post_neuron: Postsynaptic neuron
            weight: Synaptic weight
            tau: Time constant for decay (ms)
            reversal_potential: Synaptic reversal potential (mV)
        """
        super().__init__(pre_neuron, post_neuron, weight)
        self.tau = tau
        self.reversal_potential = reversal_potential  # 0 mV for excitatory, -80 mV for inhibitory

    def current(self, t, v_post):
        """Calculate synaptic current with exponential decay."""
        if not self.active or t < self.last_spike_time + self.delay:
            return 0.0

        # Time since the spike arrived at the synapse
        dt = t - (self.last_spike_time + self.delay)
        if dt < 0:
            return 0.0

        # Conductance-based model (g * (V - E))
        g = self.weight * np.exp(-dt / self.tau)
        return g * (self.reversal_potential - v_post)


class STDPSynapse(ExponentialSynapse):
    """Synapse with spike-timing-dependent plasticity (STDP)."""

    def __init__(self, pre_neuron, post_neuron, weight=0.5, tau=2.0, reversal_potential=0.0,
                 a_plus=0.1, a_minus=0.12, tau_plus=20.0, tau_minus=20.0):
        """Initialize a synapse with STDP.

        Args:
            pre_neuron: Presynaptic neuron
            post_neuron: Postsynaptic neuron
            weight: Initial synaptic weight
            tau: Time constant for current decay (ms)
            reversal_potential: Synaptic reversal potential (mV)
            a_plus: LTP learning rate
            a_minus: LTD learning rate
            tau_plus: LTP time constant (ms)
            tau_minus: LTD time constant (ms)
        """
        super().__init__(pre_neuron, post_neuron, weight, tau, reversal_potential)

        # STDP parameters
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # Trace variables for STDP
        self.pre_trace = 0.0
        self.post_trace = 0.0

        # Weight limits
        self.w_min = 0.0
        self.w_max = 5.0

        # Weight history for visualization
        self.weight_history = [(0.0, weight)]

    def update_pre_spike(self, t):
        """Update synapse on presynaptic spike.

        Args:
            t: Time of spike (ms)
        """
        # LTD: if post fired before pre
        dw = -self.a_minus * self.post_trace
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)

        # Reset pre-trace
        self.pre_trace = 1.0

        # Record weight change
        self.weight_history.append((t, self.weight))

    def update_post_spike(self, t):
        """Update synapse on postsynaptic spike.

        Args:
            t: Time of spike (ms)
        """
        # LTP: if pre fired before post
        dw = self.a_plus * self.pre_trace
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)

        # Reset post-trace
        self.post_trace = 1.0

        # Record weight change
        self.weight_history.append((t, self.weight))

    def update_traces(self, dt):
        """Update STDP traces.

        Args:
            dt: Time step (ms)
        """
        # Decay traces exponentially
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)


class CompoundSynapse(Synapse):
    """Models a compound synapse with multiple neurotransmitter components."""

    def __init__(self, pre_neuron, post_neuron, weight_ampa=0.5, weight_nmda=0.3, weight_gaba=0.0):
        """Initialize a compound synapse with multiple components.

        Args:
            pre_neuron: Presynaptic neuron
            post_neuron: Postsynaptic neuron
            weight_ampa: Weight of fast AMPA component (excitatory)
            weight_nmda: Weight of slow NMDA component (excitatory)
            weight_gaba: Weight of GABA component (inhibitory)
        """
        super().__init__(pre_neuron, post_neuron, weight=1.0)  # Weight not used directly

        # AMPA component (fast excitatory)
        self.ampa = ExponentialSynapse(pre_neuron, post_neuron, weight_ampa, tau=2.0, reversal_potential=0.0)

        # NMDA component (slow excitatory with voltage dependence)
        self.nmda = NMDASynapse(pre_neuron, post_neuron, weight_nmda)

        # GABA component (inhibitory)
        self.gaba = ExponentialSynapse(pre_neuron, post_neuron, weight_gaba, tau=6.0, reversal_potential=-80.0)

    def receive_spike(self, t):
        """Receive a spike from the presynaptic neuron."""
        self.last_spike_time = t
        self.ampa.receive_spike(t)
        self.nmda.receive_spike(t)
        self.gaba.receive_spike(t)

    def current(self, t, v_post):
        """Calculate the total current from all components."""
        i_ampa = self.ampa.current(t, v_post)
        i_nmda = self.nmda.current(t, v_post)
        i_gaba = self.gaba.current(t, v_post)

        return i_ampa + i_nmda + i_gaba


class NMDASynapse(Synapse):
    """NMDA synapse with voltage-dependent magnesium block."""

    def __init__(self, pre_neuron, post_neuron, weight=0.5, tau_rise=2.0, tau_decay=100.0):
        """Initialize NMDA synapse.

        Args:
            pre_neuron: Presynaptic neuron
            post_neuron: Postsynaptic neuron
            weight: Synaptic weight
            tau_rise: Rise time constant (ms)
            tau_decay: Decay time constant (ms)
        """
        super().__init__(pre_neuron, post_neuron, weight)
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.reversal_potential = 0.0  # Excitatory

    def mgb(self, v):
        """Magnesium block function.

        Args:
            v: Membrane potential (mV)

        Returns:
            Scaling factor (0-1) for NMDA conductance
        """
        return 1.0 / (1.0 + np.exp(-0.062 * v) * (1.0/3.57))

    def current(self, t, v_post):
        """Calculate NMDA current with voltage dependence."""
        if not self.active or t < self.last_spike_time + self.delay:
            return 0.0

        # Time since the spike arrived at the synapse
        dt = t - (self.last_spike_time + self.delay)
        if dt < 0:
            return 0.0

        # Dual exponential waveform for NMDA
        g_norm = (np.exp(-dt/self.tau_decay) - np.exp(-dt/self.tau_rise)) / \
                 (self.tau_decay - self.tau_rise)

        # Scale by weight and add voltage-dependent Mg2+ block
        g = self.weight * g_norm * self.mgb(v_post)

        # Current = conductance * driving force
        return g * (self.reversal_potential - v_post)
