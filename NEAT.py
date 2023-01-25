import neat


def evaluate_fitness(agent):
    agent.reset()
    episode_rewards = []
    while not agent.done:
        agent.step()
        episode_rewards.append(agent.reward)
    return sum(episode_rewards)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # evaluate the genome using some dataset or task
        fitness = evaluate_fitness(net)

        genome.fitness = fitness


# load the configuration file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# create the population
pop = neat.Population(config)

# add a stdout reporter to show progress in the terminal
pop.add_reporter(neat.StdOutReporter(True))

# run for up to 300 generations
winner = pop.run(eval_genomes, 300)

# display the winning genome
print(winner)
