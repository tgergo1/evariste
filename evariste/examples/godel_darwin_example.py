#!/usr/bin/env python3

import numpy as np
from evariste.models.godel_darwin import (
    GodelDarwinMachine, 
    SymbolicProcessor,
    EvolutionaryOptimizer
)
from evariste.neuron import Neuron
from evariste.models.network import Network


def simple_fitness_function(individual):
    return sum(individual.get('genes', [0]))


def main():
    print("godel-darwin machine example")
    print("-" * 40)

    gdm = GodelDarwinMachine("example_gdm")

    symbolic = SymbolicProcessor("symbolic_processor")
    evolutionary = EvolutionaryOptimizer("evolutionary_optimizer", population_size=50)

    symbolic.add_axiom("All humans are mortal")
    symbolic.add_axiom("Socrates is human")

    evolutionary.set_fitness_function(simple_fitness_function)
    evolutionary.initialize_population({"genes": np.random.rand(10)})

    gdm.add_module(symbolic)
    gdm.add_module(evolutionary)

    gdm.connect("symbolic_processor", "theorems", "evolutionary_optimizer", "constraints")

    inputs = {
        "symbolic_processor": {"query": "Is Socrates mortal?"},
        "evolutionary_optimizer": {"target": 0.8}
    }

    outputs = gdm.forward(inputs)

    print("\nsymbolic processor output:")
    for key, value in outputs["symbolic_processor"].items():
        print(f"  {key}: {value}")

    print("\nevolutionary optimizer output:")
    for key, value in outputs["evolutionary_optimizer"].items():
        if key == "best_individual":
            print(f"  {key}: {type(value).__name__} with genes shape {np.array(value['genes']).shape}")
        else:
            print(f"  {key}: {value}")

    network = Network("gdm_network")

    input_neuron = Neuron("input1")
    hidden_neuron = Neuron("hidden1")
    output_neuron = Neuron("output1")

    network.add_neuron(input_neuron, layer_type="input")
    network.add_neuron(hidden_neuron, layer_type="hidden")
    network.add_neuron(output_neuron, layer_type="output")

    network.connect("input1", "hidden1", weight=0.5)
    network.connect("hidden1", "output1", weight=0.8)

    neural_outputs = network.forward({"input1": 1.0})
    print("\nneural network output:")
    for neuron_id, output in neural_outputs.items():
        print(f"  {neuron_id}: {output}")

    integrated_output = gdm.integrate_with_neural_network(neural_outputs)
    print("\nintegrated output:")
    print(f"  {integrated_output}")


if __name__ == "__main__":
    main()
