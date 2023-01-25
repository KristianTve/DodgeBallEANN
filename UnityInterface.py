import sys

import neat
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import ActionTuple  # Creating a compatible action
import numpy as np
import atexit

env = UE(seed=1, side_channels=[])
env.reset()  # Resets the environment ready for the next simulation
show_prints = True
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
generation = 0

if show_prints:
    print(f"Name of the behavior : {behavior_name}")
    print("Number of observations : ", len(spec.observation_specs))
    print(spec.observation_specs[0].observation_type)


# Handles the exit by closing the unity environment to avoid _communicator errors.
def exit_handler():
    print("EXITING")
    env.close()


atexit.register(exit_handler)


def run_agent(genomes, config):
    """
    Population size is configured as 12 to suit the training environment!
    :param genomes: All the genomes in the current generation.
    :param config: Configuration files
    :return: Best genome from generation.
    """
    # TODO NEAT REQUIRE RUN TO BE ONE EPISODE ONLY
    # Decision Steps is a list of all agents requesting a decision
    # Terminal steps is all agents that has reached a terminal state (finished)
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Empty array to save all the neural networks for the (two) agents
    # TODO Should two networks from the same generation play against each other?
    # TODO Otherwise a solution can be picked randomly to play against and only implement logic for one agent here.
    policies = []

    # Initialize the neural networks for each genome.
    for i, g in genomes:
        # Each agent has their own genome which denotes their phenotype
        # Genomes consists of properties:
        # Key (ID)
        # Fitness (score)
        # Nodes and connections
        policy = neat.nn.FeedForwardNetwork.create(g, config)
        policies.append(policy)
        g.fitness = 0

    if show_prints:
        print(list(decision_steps))
        print(list(terminal_steps))

    global generation
    generation += 1
    done = False  # For the tracked_agent
    total_reward = 0

    # Agents:
    agent_count = len(decision_steps.agent_id)  #
    removed_agents = []

    while not done:
        agents = list(decision_steps)  # Agent IDs that are alive

        # Store actions for each agent with 5 actions per agent (3 continuous and 2 discrete)
        actions = np.zeros(shape=(23, 5))  # 23 in size because of the agent IDs going up to 22.

        # Concatenate all the data BESIDES number 3 (OtherAgentsData)
        nn_input = np.zeros(shape=(23, 364))  # 23 in size because of the agent IDs going up to 22.

        for agent in agents:  # Collect observations from the agents requesting input
            nn_input[agent] = np.concatenate((decision_steps[agent].obs[0],
                                              decision_steps[agent].obs[1],
                                              decision_steps[agent].obs[3],
                                              decision_steps[agent].obs[4],
                                              decision_steps[agent].obs[5]))

        if len(decision_steps) > 0:
            for agent_index in agents:
                if agent_index in decision_steps:
                    if agent_index != 0:    # Avoid divide by 0
                        action = policies[int(agent_index/2)].activate(nn_input[agent_index])  # Since its only partall
                    else:
                        action = policies[agent_index].activate(nn_input[agent_index])

                    actions[agent_index] = action

        # Clip discrete values to 0 or 1
        for agent in agents:
            actions[agent, 3] = 1 if actions[agent, 3] > 0 else 0   # Shoot
            actions[agent, 4] = 1 if actions[agent, 4] > 0 else 0   # DASH


        # Set actions for each agent (convert from ndarray to ActionTuple)
        if len(decision_steps.agent_id) != 0:
            for agent in agents:
                continuous_actions = [actions[agent, 0:3]]
                discrete_actions = [actions[agent, 3:5]]
                action_tuple = ActionTuple(discrete=np.array(discrete_actions), continuous=np.array(continuous_actions))
                env.set_action_for_agent(behavior_name=behavior_name, agent_id=agent, action=action_tuple)

        # Move the simulation forward
        env.step()

        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        reward = 0
        # Collect reward
        for agent_index in agents:
            if agent_index in decision_steps:  # The agent requested a decision
                reward += decision_steps[agent_index].reward
            elif agent_index in terminal_steps:
                reward += terminal_steps[agent_index].reward

            if agent_index != 0:
                genomes[int(agent_index/2)][1].fitness += reward
            else:
                genomes[int(agent_index)][1].fitness += reward
            total_reward += reward  # Testing purposes

        # When all agents has reached a terminal state, then the game is over
        if len(decision_steps) == 0:
            done = True

        # Reward status
        sys.stdout.write("\rReward: %i" % total_reward)
        sys.stdout.flush()

    # Clean the environment for a new generation.
    env.reset()
    print("Finished generation")


if __name__ == "__main__":
    # Set configuration file
    config_path = "./config"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    best_genome = p.run(run_agent, 2)

