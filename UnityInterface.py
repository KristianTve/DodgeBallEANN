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

    # Decision_steps.obs prints out the raw observations of the agent.
    # print("DECISION STEPS OBSERVATIONS")
    # print(decision_steps.obs)  # Observations from the agent(s) state
    # print("DECISION STEPS OBS END")


def exit_handler():
    print("EXITING")
    env.close()


atexit.register(exit_handler)


def run_agent(genomes, config):
    """
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

    while not done:
        """
        print("OBSERVATION 1")
        print(decision_steps[agent_index].obs[0])   # Agent RayCast
        print("OBSERVATION 2")
        print(decision_steps[agent_index].obs[1])   # Ball RayCast
        print("OBSERVATION 3")
        print(decision_steps[agent_index].obs[2])   # Other Agents
        print("OBSERVATION 4")
        print(decision_steps[agent_index].obs[3])   # Rear Sensor
        print("OBSERVATION 5")
        print(decision_steps[agent_index].obs[4])   # Params Collected
        print("OBSERVATION 6")  
        print(decision_steps[agent_index].obs[5])   # Wall RayCast
        """

        # Concatenate all the data BESIDES number 3 (OtherAgentsData)
        nn_input = np.concatenate((decision_steps[0].obs[0],
                                   decision_steps[0].obs[1],
                                   decision_steps[0].obs[3],
                                   decision_steps[0].obs[4],
                                   decision_steps[0].obs[5]))

        action = policies[0].activate(nn_input)  # Works
        action = np.array(action)  # Convert to ndarray

        # Not elegant lol
        action[4] = 1 if action[4] > 0 else 0       # DASH
        action[5] = 1 if action[5] > 0 else 0       # THROW

        #action[4] = 0
        #action[5] = 1

        #print("ACTIONS CHOSEN:")
        #print(action)

        # Set the actions
        action_tuple = ActionTuple(continuous=np.array([action[0:3]]), discrete=np.array([action[3:5]]))
        #env.set_actions(behavior_name, action_tuple)

        action = spec.action_spec.random_action(len(decision_steps))
        action.add_continuous(np.array([[0.0, 0.0, 0.0]]))
        env.set_actions(behavior_name, action)

        # Move the simulation forward
        env.step()
        # Get the new simulation results

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if 0 in decision_steps:  # The agent requested a decision
            total_reward += decision_steps[0].reward
        if 0 in terminal_steps:  # The agent terminated its episode
            total_reward += terminal_steps[0].reward
            done = True

        sys.stdout.write("\rReward: %i" % total_reward)
        sys.stdout.flush()

        if total_reward > 0.0:
            print("REWARD MANs")
            print(total_reward)

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
