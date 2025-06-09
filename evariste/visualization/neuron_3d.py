"""3D visualization of biological neurons.

This module provides tools for 3D visualization of detailed neuron morphology
including dendrites, soma, and axons with activity visualization.
"""
import random

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import colorsys


class NeuronMesh:
    """Create 3D mesh for a neuron with realistic morphology."""

    def __init__(self, neuron):
        """Initialize neuron mesh generator.

        Args:
            neuron: HodgkinHuxleyNeuron object
        """
        self.neuron = neuron
        self.soma_mesh = None
        self.dendrite_meshes = []
        self.axon_meshes = []

        # Generate the mesh components
        self.create_soma_mesh()
        self.create_dendrite_meshes()
        self.create_axon_meshes()

    def create_soma_mesh(self):
        """Create a sphere mesh for the soma."""
        md = gl.MeshData.sphere(rows=10, cols=10, radius=self.neuron.soma_radius)
        mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded')

        # Position at the neuron's location
        mesh.translate(self.neuron.position[0], self.neuron.position[1], self.neuron.position[2])

        self.soma_mesh = mesh

    def create_dendrite_meshes(self):
        """Create cylinder meshes for dendrites."""
        for dendrite in self.neuron.dendrites:
            start = dendrite['start']
            end = dendrite['end']
            diameter = dendrite['diameter']

            # Create a cylinder along the path
            # Calculate direction vector and length
            direction = np.array(end) - np.array(start)
            length = np.linalg.norm(direction)
            if length == 0:
                continue

            # Create cylinder aligned with z-axis
            radius = diameter/2
            md = gl.MeshData.cylinder(rows=10, cols=10, radius=[radius, radius], length=length)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded')

            # Rotate and position the cylinder
            # First normalize the direction
            direction = direction / length

            # Find rotation axis and angle to align z-axis with direction
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                dot_product = np.dot(z_axis, direction)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                # Convert to degrees and set rotation
                angle_deg = angle * 180 / np.pi
                mesh.rotate(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])

            # Position at the start point
            mesh.translate(start[0], start[1], start[2])

            self.dendrite_meshes.append(mesh)

    def create_axon_meshes(self):
        """Create cylinder meshes for axon segments."""
        for axon_segment in self.neuron.axon:
            start = axon_segment['start']
            end = axon_segment['end']
            diameter = axon_segment['diameter']

            # Similar to dendrite mesh creation
            direction = np.array(end) - np.array(start)
            length = np.linalg.norm(direction)
            if length == 0:
                continue

            radius = diameter/2
            md = gl.MeshData.cylinder(rows=10, cols=10, radius=[radius, radius], length=length)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded')

            # Rotate and position
            direction = direction / length
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                dot_product = np.dot(z_axis, direction)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_deg = angle * 180 / np.pi
                mesh.rotate(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])

            mesh.translate(start[0], start[1], start[2])

            self.axon_meshes.append(mesh)

    def update_colors(self):
        """Update mesh colors based on neuron state."""
        # Map membrane potential to color
        v = self.neuron.v
        # Normalize between -80 and +40 mV
        v_norm = (v + 80) / 120  # 0 to 1 range
        v_norm = np.clip(v_norm, 0, 1)

        # Create color gradient: blue (-80mV) to red (+40mV)
        # Using HSV: blue (240°) to red (0°)
        hue = (1 - v_norm) * 0.7  # 0.7 = 240° in 0-1 range
        saturation = 0.9
        value = 0.9

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        # Brighter color during spikes
        if self.neuron.v > 0:  # Approximation of spike
            value = 1.0
            r, g, b = 1.0, 1.0, 0.5  # Yellow flash

        # Set colors for all components
        if self.soma_mesh:
            self.soma_mesh.setColor(QColor(int(r*255), int(g*255), int(b*255)))

        for mesh in self.dendrite_meshes:
            mesh.setColor(QColor(int(r*255), int(g*255), int(b*255)))

        for mesh in self.axon_meshes:
            mesh.setColor(QColor(int(r*255), int(g*255), int(b*255)))

    def add_to_view(self, view):
        """Add all mesh components to the 3D view.

        Args:
            view: GLViewWidget to add items to
        """
        if self.soma_mesh:
            view.addItem(self.soma_mesh)

        for mesh in self.dendrite_meshes:
            view.addItem(mesh)

        for mesh in self.axon_meshes:
            view.addItem(mesh)

    def remove_from_view(self, view):
        """Remove all mesh components from the 3D view."""
        if self.soma_mesh:
            view.removeItem(self.soma_mesh)

        for mesh in self.dendrite_meshes:
            view.removeItem(mesh)

        for mesh in self.axon_meshes:
            view.removeItem(mesh)


class NetworkVisualizer(QMainWindow):
    """Main window for 3D network visualization."""

    def __init__(self, network):
        """Initialize the network visualizer.

        Args:
            network: BiologicalNeuralNetwork object
        """
        super().__init__()
        self.network = network
        self.neuron_meshes = []  # List of NeuronMesh objects
        self.synapse_lines = []  # List of GLLinePlotItem objects
        self.voltage_plots = {}  # Dictionary of neuron_id -> PlotItem

        self.init_ui()
        self.create_neuron_meshes()
        self.create_synapse_visualizations()

        # Setup timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(50)  # Update every 50 ms

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Biological Neural Network Visualization")
        self.resize(1200, 800)

        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)

        # 3D view on the left
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=400)

        # Add coordinate grid
        grid = gl.GLGridItem()
        self.view3d.addItem(grid)

        # Add X, Y, Z axes
        for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], 
                                          [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])):
            axis_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], 
                                                       [100 if i==0 else 0, 
                                                        100 if i==1 else 0, 
                                                        100 if i==2 else 0]]), 
                                        color=color, width=2)
            self.view3d.addItem(axis_line)

            # Add axis label
            axis_label = gl.GLTextItem(pos=np.array([110 if i==0 else 0, 
                                                   110 if i==1 else 0, 
                                                   110 if i==2 else 0]), 
                                    text=axis)
            self.view3d.addItem(axis_label)

        splitter.addWidget(self.view3d)

        # Right panel with voltage plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Add title for plots
        right_layout.addWidget(QLabel("Membrane Potential Plots"))

        # Create plot widget for voltage traces
        self.plot_widget = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.plot_widget)

        splitter.addWidget(right_panel)

        # Set splitter as central widget
        self.setCentralWidget(splitter)

        # Initialize voltage plots for a few neurons
        self.init_voltage_plots()

    def init_voltage_plots(self, max_plots=5):
        """Initialize voltage plots for a subset of neurons.

        Args:
            max_plots: Maximum number of plots to create
        """
        num_plots = min(max_plots, len(self.network.neurons))

        for i in range(num_plots):
            plot = self.plot_widget.addPlot(row=i, col=0)
            plot.setLabel('left', f"Neuron {i}")
            plot.setLabel('bottom', 'Time (ms)')
            plot.setYRange(-80, 40)  # Typical range for membrane potential

            # Add plot curve
            curve = plot.plot(pen=pg.mkPen(color=(i*50 % 255, 255 - i*30 % 255, 100 + i*20 % 155)))

            self.voltage_plots[i] = {
                'plot': plot,
                'curve': curve,
                'data_x': [],
                'data_y': []
            }

    def create_neuron_meshes(self):
        """Create mesh visualizations for all neurons."""
        # First, check if neurons have morphology defined
        # If not, add some basic morphology
        for neuron in self.network.neurons:
            # Add basic morphology if none exists
            if not hasattr(neuron, 'soma_radius'):
                neuron.soma_radius = 5.0

            if not hasattr(neuron, 'dendrites') or not neuron.dendrites:
                # Add some random dendrites
                neuron.dendrites = []
                pos = np.array(neuron.position)

                # Add 3-5 dendrites
                for _ in range(random.randint(3, 5)):
                    # Random direction unit vector
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)

                    # Length between 20-50 units
                    length = random.uniform(20, 50)
                    end = pos + direction * length

                    # Diameter between 0.5-2.0 units
                    diameter = random.uniform(0.5, 2.0)

                    neuron.add_dendrite(pos, end, diameter)

            if not hasattr(neuron, 'axon') or not neuron.axon:
                # Add a simple axon
                neuron.axon = []
                pos = np.array(neuron.position)

                # Random direction for axon
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)

                # Axon length (longer than dendrites)
                length = random.uniform(60, 100)
                end = pos + direction * length

                neuron.add_axon_segment(pos, end, diameter=0.8)

        # Create mesh for each neuron
        for neuron in self.network.neurons:
            mesh = NeuronMesh(neuron)
            mesh.add_to_view(self.view3d)
            self.neuron_meshes.append(mesh)

    def create_synapse_visualizations(self):
        """Create visual representations of synapses."""
        for synapse in self.network.synapses:
            pre_pos = np.array(synapse.pre_neuron.position)
            post_pos = np.array(synapse.post_neuron.position)

            # Find an endpoint on the axon to connect from
            if synapse.pre_neuron.axon:
                pre_pos = np.array(synapse.pre_neuron.axon[-1]['end'])

            # Find an endpoint on a dendrite to connect to
            if synapse.post_neuron.dendrites:
                # Find closest dendrite endpoint
                dendrite_ends = [np.array(d['end']) for d in synapse.post_neuron.dendrites]
                distances = [np.linalg.norm(pre_pos - end) for end in dendrite_ends]
                closest_idx = np.argmin(distances)
                post_pos = dendrite_ends[closest_idx]

            # Create a line for the synapse
            line_points = np.array([pre_pos, post_pos])

            # Determine color based on synapse type
            if hasattr(synapse, 'reversal_potential'):
                # Excitatory (reversal_potential near 0) is green, inhibitory (negative) is red
                if synapse.reversal_potential > -40:  # Excitatory
                    color = (0, 1, 0, 0.7)  # Green
                else:  # Inhibitory
                    color = (1, 0, 0, 0.7)  # Red
            else:
                color = (0.7, 0.7, 0.7, 0.7)  # Gray default

            # Create line item
            line = gl.GLLinePlotItem(pos=line_points, color=color, width=1)
            self.view3d.addItem(line)
            self.synapse_lines.append(line)

    def update_visualization(self):
        """Update visualization based on current network state."""
        # Update neuron colors based on membrane potentials
        for mesh in self.neuron_meshes:
            mesh.update_colors()

        # Update voltage plots
        current_time = self.network.time
        for neuron_id, plot_data in self.voltage_plots.items():
            if neuron_id < len(self.network.neurons):
                neuron = self.network.neurons[neuron_id]

                # Add new data point
                plot_data['data_x'].append(current_time)
                plot_data['data_y'].append(neuron.v)

                # Limit data points to keep performance reasonable
                max_points = 1000
                if len(plot_data['data_x']) > max_points:
                    plot_data['data_x'] = plot_data['data_x'][-max_points:]
                    plot_data['data_y'] = plot_data['data_y'][-max_points:]

                # Update plot curve
                plot_data['curve'].setData(plot_data['data_x'], plot_data['data_y'])

                # Update x-range to show recent activity
                if current_time > 100:
                    plot_data['plot'].setXRange(current_time - 100, current_time)

        # Update synapse lines (could animate signal propagation)
        for i, synapse in enumerate(self.network.synapses):
            if i < len(self.synapse_lines):
                # Check if a spike is propagating through this synapse
                time_since_spike = current_time - synapse.last_spike_time
                if 0 <= time_since_spike <= synapse.delay:
                    # Spike is propagating - make the line brighter
                    progress = time_since_spike / synapse.delay

                    # Get current color
                    current_color = self.synapse_lines[i].color

                    # Change width and brightness during spike propagation
                    width = 3 * (1 - progress) + 1  # Thicker at start, thinner at end

                    # Make a bright pulse traveling along the line
                    pulse_pos = progress
                    pre_pos = np.array(synapse.pre_neuron.position)
                    post_pos = np.array(synapse.post_neuron.position)
                    mid_point = pre_pos + (post_pos - pre_pos) * pulse_pos

                    # Create a temporary bright point at the pulse position
                    # This is a simplified visualization - in a more advanced version,
                    # you could create a small bright sphere that moves along the synapse

                    # Update the line width
                    self.synapse_lines[i].setWidth(width)
                else:
                    # Normal state
                    self.synapse_lines[i].setWidth(1)


def generate_morphology(neuron):
    """Generate realistic morphology for a neuron.

    Args:
        neuron: HodgkinHuxleyNeuron object
    """
    # Set soma radius
    neuron.soma_radius = random.uniform(8.0, 12.0)

    # Generate dendrites using a recursive branching pattern
    def generate_dendrite_branch(start_point, direction, length, diameter, branch_level=0):
        # End point of this branch
        direction = direction / np.linalg.norm(direction)  # Normalize
        end_point = start_point + direction * length

        # Add this branch as a dendrite segment
        neuron.add_dendrite(start_point, end_point, diameter)

        # Stop branching after level 3
        if branch_level >= 3:
            return

        # Branch with some probability
        n_branches = random.randint(0, 2)  # 0, 1, or 2 new branches
        for _ in range(n_branches):
            # New direction is a perturbation of current direction
            new_direction = direction + np.random.randn(3) * 0.3
            new_direction = new_direction / np.linalg.norm(new_direction)

            # New branches are shorter and thinner
            new_length = length * random.uniform(0.6, 0.8)
            new_diameter = diameter * random.uniform(0.7, 0.9)

            # Recursive branching
            generate_dendrite_branch(end_point, new_direction, new_length, new_diameter, branch_level+1)

    # Generate 3-5 primary dendrites
    for _ in range(random.randint(3, 5)):
        # Random direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)

        # Primary dendrite parameters
        length = random.uniform(30, 50)
        diameter = random.uniform(1.0, 2.0)

        # Generate branches recursively
        generate_dendrite_branch(np.array(neuron.position), direction, length, diameter)

    # Generate axon with multiple segments
    # Start at the soma
    current_point = np.array(neuron.position)

    # Pick a primary direction for the axon
    axon_direction = np.random.randn(3)
    axon_direction = axon_direction / np.linalg.norm(axon_direction)

    # Generate 3-5 axon segments
    axon_diameter = random.uniform(0.8, 1.2)
    for _ in range(random.randint(3, 5)):
        # Each segment continues roughly in the same direction with some randomness
        direction = axon_direction + np.random.randn(3) * 0.2
        direction = direction / np.linalg.norm(direction)

        # Length of this segment
        length = random.uniform(40, 70)
        end_point = current_point + direction * length

        # Add the segment
        neuron.add_axon_segment(current_point, end_point, axon_diameter)

        # Update for next segment
        current_point = end_point
        axon_direction = direction
        axon_diameter *= random.uniform(0.9, 1.0)  # Slight thinning


def create_visualization_app(network):
    """Create and run the 3D visualization application.

    Args:
        network: BiologicalNeuralNetwork object

    Returns:
        QApplication instance and NetworkVisualizer window
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Generate morphology for neurons if needed
    for neuron in network.neurons:
        generate_morphology(neuron)

    # Create the visualization window
    visualizer = NetworkVisualizer(network)
    visualizer.show()

    return app, visualizer
