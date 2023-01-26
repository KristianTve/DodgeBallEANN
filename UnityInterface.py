import sys

import neat
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import ActionTuple  # Creating a compatible action
import numpy as np
import atexit

env = UE(seed=1, side_channels=[])
env.reset()  # Resets the environment ready for the next simulation
show_prints = True
behavior_name_purple = list(env.behavior_specs)[0]
behavior_name_blue = list(env.behavior_specs)[1]

spec_purple = env.behavior_specs[behavior_name_purple]
spec_blue = env.behavior_specs[behavior_name_blue]

generation = 0

if show_prints:
    print(f"Name of the behavior : {behavior_name_purple}")
    print("Number of observations : ", len(spec_purple.observation_specs))
    print(spec_purple.observation_specs[0].observation_type)

    print(f"Name of the behavior for players : {behavior_name_blue}")
    print("Number of observations : ", len(spec_blue.observation_specs))
    print(spec_blue.observation_specs[0].observation_type)


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
    # Decision Steps is a list of all agents requesting a decision
    # Terminal steps is all agents that has reached a terminal state (finished)
    decision_steps_purple, terminal_steps_purple = env.get_steps(behavior_name_purple)
    decision_steps_blue, terminal_steps_blue = env.get_steps(behavior_name_blue)

    # TODO Implement a option to run a given neural network for all agents of one team, of which learning is disabled.
    # TODO But that requires only 12 players on one team.
    # Empty array to save all the neural networks for all agents on both teams
    policies = []

    # Initialize the neural networks for each genome.
    for i, g in genomes:
        # Each agent has their own genome which denotes their phenotype
        # Genomes consists of properties:
        # Key (ID)
        # Fitness (score)
        # Nodes and connections
        # i starts at 1
        policy = neat.nn.FeedForwardNetwork.create(g, config)
        policies.append(policy)
        g.fitness = 0

    global generation
    generation += 1
    done = False  # For the tracked_agent
    total_reward = 0

    # Agents:
    agent_count_purple = len(decision_steps_purple.agent_id)  # 12
    agent_count_blue = len(decision_steps_blue.agent_id)  # 12
    agent_count = agent_count_purple + agent_count_blue  # 24

    removed_agents = []

    while not done:
        agents_purple = list(decision_steps_purple)  # Agent IDs that are alive
        agents_blue = list(decision_steps_blue)  # Agent IDs that are alive

        # Store actions for each agent with 5 actions per agent (3 continuous and 2 discrete)
        actions = np.zeros(shape=(24, 5))  # 23 in size because of the agent IDs going up to 22.

        # Concatenate all the data BESIDES number 3 (OtherAgentsData)
        nn_input = np.zeros(shape=(24, 364))  # 23 in size because of the agent IDs going up to 22.

        for agent in range(agent_count):  # Collect observations from the agents requesting input
            decision_steps_nn = []
            if agent % 2 == 0 or agent == 0:
                decision_steps_nn = decision_steps_purple  # Purple agent
            else:
                decision_steps_nn = decision_steps_blue  # Blue agent

            nn_input[agent] = np.concatenate((decision_steps_nn[agent].obs[0],
                                              decision_steps_nn[agent].obs[1],
                                              decision_steps_nn[agent].obs[3],
                                              decision_steps_nn[agent].obs[4],
                                              decision_steps_nn[agent].obs[5]))

        # Checks if the
        if (len(decision_steps_purple) > 0) and (len(decision_steps_blue) > 0):  # More steps to take?
            for agent_index in range(agent_count):  # Iterates through all the agent indexes
                if (agent_index in decision_steps_purple) or (agent_index in decision_steps_blue):  # Is agent ready?
                    action = policies[agent_index].activate(nn_input[agent_index])  # FPass for purple action
                    actions[agent_index] = action  # Save action in array of actions

        # Clip discrete values to 0 or 1
        for agent in range(agent_count):
            actions[agent, 3] = 1 if actions[agent, 3] > 0 else 0  # Shoot
            actions[agent, 4] = 1 if actions[agent, 4] > 0 else 0  # DASH

        # Set actions for each agent (convert from ndarray to ActionTuple)
        if len(decision_steps_purple.agent_id) != 0 and len(decision_steps_blue.agent_id) != 0:
            for agent in range(agent_count):
                # Creating a action tuple
                continuous_actions = [actions[agent, 0:3]]
                discrete_actions = [actions[agent, 3:5]]
                action_tuple = ActionTuple(discrete=np.array(discrete_actions), continuous=np.array(continuous_actions))

                # Applying the action to respective agents on both teams
                if agent % 2 == 0 or agent == 0:
                    env.set_action_for_agent(behavior_name=behavior_name_purple, agent_id=agent, action=action_tuple)
                else:
                    env.set_action_for_agent(behavior_name=behavior_name_blue, agent_id=agent, action=action_tuple)

        # Move the simulation forward
        env.step()

        # Get the new simulation results
        decision_steps_purple, terminal_steps_purple = env.get_steps(behavior_name_purple)
        decision_steps_blue, terminal_steps_blue = env.get_steps(behavior_name_blue)

        # Collect reward
        reward = 0
        for agent_index in range(agent_count):
            if agent_index % 2 == 0 or agent_index == 0:
                if agent_index in decision_steps_purple:  # The agent requested a decision
                    reward += decision_steps_purple[agent_index].reward
                elif agent_index in terminal_steps_purple:
                    reward += terminal_steps_purple[agent_index].reward
            else:
                if agent_index in decision_steps_blue:  # The agent requested a decision
                    reward += decision_steps_blue[agent_index].reward
                elif agent_index in terminal_steps_blue:
                    reward += terminal_steps_blue[agent_index].reward

            genomes[agent_index][1].fitness += reward
            total_reward += reward  # Testing purposes

        # When whole teams are eliminated, end the generation.
        if len(decision_steps_blue) == 0 or len(decision_steps_purple) == 0:
            done = True

        # Reward status
        sys.stdout.write("\rCollective reward: %d | Blue left: %d | Purple left: %d" % (total_reward,
                                                                                        len(decision_steps_blue),
                                                                                        len(decision_steps_purple)))
        sys.stdout.flush()

    # Clean the environment for a new generation.
    env.reset()
    print("Finished generation")


if __name__ == "__main__":
    load_from_checkpoint = True

    # Set configuration file
    config_path = "./config"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    if load_from_checkpoint:  # Load from checkpoint
        p = neat.Checkpointer.restore_checkpoint("NEAT-checkpoint-3")
        print("LOADED FROM CHECKPOINT")
    else:   # Or generate new initial population
        p = neat.Population(config)

    # For saving checkpoints during training
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='checkpoints/NEAT-checkpoint-'))

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    best_genome = p.run(run_agent, 50)
