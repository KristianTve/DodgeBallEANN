import random
import asyncio  # ASYNC Library

import pygad.gann
import pygad.nn

from Individual import Individual

"""
CALLBACK FUNCTINOS OF THE PYGAD LIBRARY
"""


def on_start(ga_instance):
    print("on_start()")


def on_fitness(ga_instance, population_fitness):
    print(population_fitness)  # Does this return the right thing? In example ga_instance is #1 param
    print("on_fitness()")


def on_parents(ga_instance, selected_parents):
    print("on_parents()")


def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")


def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")


def on_generation(ga_instance):
    print("on_generation()")
    populartion_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                             population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=populartion_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


def end_of_game(self):
    """
    This is called by the C# script to signal end of game.
    :return:
    """
    pass


def fitness_function(self, solution):
    """
    Evaluates the neural networks based on game performance.
    Once the game ends, the C# scripts sends the game outcome per individual to this script.

    Things to measure for success:
    - Win or loss? Or ratio of player hearts to the opponent hearts.
      So if -6 for complete loss and 6 for complete win.
    -
    :param GameState (struct) Game result for the player
    :return: (float) fitness score
    """

    return random.random()


"""
Class for the genetic algorithm optimizing neural networks.
Using the PyGAD package for the basic genetic algorithm and implementing
custom fitness function, crossover etc.
"""

"""Constructor"""
pop_size = 5  # Min 2
parents_for_crossover = 2
fitness_threshold = 200
max_generations = 5
individuals = []  # Up for debate how this is saved
state_size = 4
num_actions = 2

GANN_instance = pygad.gann.GANN(num_solutions=pop_size,
                                num_neurons_input=state_size,
                                num_neurons_hidden_layers=[3],
                                num_neurons_output=num_actions,
                                hidden_activations=["relu"],
                                output_activation="softmax")

# Fetching the population weights as vectors
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

initial_population = population_vectors.copy()  # Input for the standard GA
num_parents_mating = 2  # How many parents used for mating
num_generations = 100
mutation_percent_genes = 5  # ??
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
keep_parents = 1

init_range_low = -2
init_range_high = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_function,
                       mutation_percent_genes=mutation_percent_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,

                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop)


def algorithm():
    ga_instance.run()
    # INITIALIZE POPULATION
    # for i in range(individuals_in_population):
    #    individuals.append(Individual(state_size=2, output_size=2))

    # EVALUATE EACH CANDIDATE WITH THE FITNESS FUNCTION
    """ Send individuals to game for evaluation then evaluate fitness """

    # WHILE CONDITION NOT MET
    # SELECT PARENTS
    # RECOMBINE PAIRS OF PARENTS
    # MUTATE the resulting offspring
    # EVALUATE new candidates
    # SELECT individuals for the next generation
    # END
