#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Evariste - A bold approach to general artificial intelligence through
biologically accurate neural simulation.

This is the main entry point for the Evariste system.
"""

import argparse
import sys
import os


def main():
    """Main entry point for Evariste."""
    parser = argparse.ArgumentParser(description="Evariste Biological Neural Simulation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Neuron simulation command
    neuron_parser = subparsers.add_parser("neuron", help="Simulate a single neuron")
    neuron_parser.add_argument("--model", choices=["hodgkin-huxley", "izhikevich"], 
                             default="hodgkin-huxley", help="Neuron model type")
    neuron_parser.add_argument("--duration", type=float, default=100.0,
                             help="Simulation duration (ms)")
    neuron_parser.add_argument("--current", type=float, default=10.0,
                             help="Injection current (nA)")
    neuron_parser.add_argument("--animate", action="store_true",
                             help="Show animated visualization")

    # Network simulation command
    network_parser = subparsers.add_parser("network", help="Simulate a neural network")
    network_parser.add_argument("--type", choices=["random", "small-world", "layers"],
                              default="random", help="Network topology type")
    network_parser.add_argument("--neurons", type=int, default=20,
                              help="Number of neurons")
    network_parser.add_argument("--duration", type=float, default=500.0,
                              help="Simulation duration (ms)")
    network_parser.add_argument("--visualize", action="store_true",
                              help="Show 3D visualization")
    network_parser.add_argument("--analysis", action="store_true",
                              help="Generate analysis plots")

    # Editor command
    editor_parser = subparsers.add_parser("editor", help="Launch neural network editor")
    editor_parser.add_argument("--model", help="Path to model file to edit")

    # Load and run command
    run_parser = subparsers.add_parser("run", help="Load and run a saved network model")
    run_parser.add_argument("--model", required=True, help="Path to model file to run")
    run_parser.add_argument("--duration", type=float, default=1000.0,
                          help="Simulation duration (ms)")
    run_parser.add_argument("--visualize", action="store_true",
                          help="Show 3D visualization")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "neuron":
        # Run neuron simulation
        print(f"Starting Evariste Neuron Simulation (model: {args.model})...")
        if args.model == "hodgkin-huxley":
            if args.animate:
                from evariste.examples.hodgkin_huxley_demo import animated_simulation
                animated_simulation()
            else:
                from evariste.examples.hodgkin_huxley_demo import single_neuron_simulation
                neuron, times, voltages = single_neuron_simulation()
        elif args.model == "izhikevich":
            print("Izhikevich model not yet implemented")
            return 1

    elif args.command == "network":
        # Run network simulation
        print(f"Starting Evariste Network Simulation (type: {args.type})...")
        from evariste.examples.neural_network_demo import create_demo_network, run_simulation
        from evariste.examples.neural_network_demo import generate_analysis_plots

        # Map argument type to function parameter
        network_type_map = {
            "random": "random",
            "small-world": "small_world",
            "layers": "layers"
        }

        # Create and run network
        network = create_demo_network(network_type_map[args.type])
        app = run_simulation(network, duration=args.duration, visualize_3d=args.visualize)

        # Generate analysis if requested
        if args.analysis:
            generate_analysis_plots(network)

        # Keep app running if visualization is enabled
        if args.visualize and app is not None:
            return app.exec_()

    elif args.command == "editor":
        # Run the editor
        print("Starting Evariste Network Editor...")
        try:
            from evariste.editor.editor import run
            return run(args.model)
        except ImportError:
            print("Editor module not fully implemented yet")
            return 1

    elif args.command == "run":
        # Load and run saved model
        print(f"Loading and running model from {args.model}...")

        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found")
            return 1

        try:
            from evariste.models.network import BiologicalNeuralNetwork
            network = BiologicalNeuralNetwork()
            network.load_state(args.model)

            from evariste.examples.neural_network_demo import run_simulation
            app = run_simulation(network, duration=args.duration, 
                               visualize_3d=args.visualize)

            if args.visualize and app is not None:
                return app.exec_()
        except Exception as e:
            print(f"Error loading or running model: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Evariste - A bold approach to general artificial intelligence

This is the main entry point for the Evariste system.
"""

import argparse
import sys

from evariste.trainer import trainer
from evariste.editor import editor
from evariste.skeleton import skeleton


def main():
    parser = argparse.ArgumentParser(description="Evariste AI System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Trainer command
    trainer_parser = subparsers.add_parser("train", help="Train a system of lobes")
    trainer_parser.add_argument("--config", help="Path to configuration file")

    # Editor command
    editor_parser = subparsers.add_parser("edit", help="Edit system of lobes")
    editor_parser.add_argument("--model", help="Path to model file to edit")

    # Skeleton command
    skeleton_parser = subparsers.add_parser("run", help="Run a trained model")
    skeleton_parser.add_argument("--model", required=True, help="Path to model file to run")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        # Run the trainer
        print("Starting Evariste Trainer...")
        # trainer.run(args.config)
    elif args.command == "edit":
        # Run the editor
        print("Starting Evariste Editor...")
        # editor.run(args.model)
    elif args.command == "run":
        # Run the skeleton
        print("Starting Evariste Skeleton...")
        # skeleton.run(args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
