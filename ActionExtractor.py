from multiprocessing import Queue
import multiprocessing as mp
import time


def get_action(network, obs, agent_num, queue):
    queue.put([agent_num, network.activate(obs)])


def get_actions(policies,
                fixed_policy,
                fixed_opponent,
                nn_input,
                decision_steps_blue,
                decision_steps_purple,
                agent_count,
                local_to_agent_map):
    # Concurrency things
    # print("CPU Cores: " + str(num_workers))
    pool = mp.Pool(processes=4)  # Problem: Unity connection (MLAgents) being duped
    q = Queue()
    for agent in range(agent_count):
        if local_to_agent_map[agent] in decision_steps_purple or local_to_agent_map[agent] in decision_steps_blue:
            if local_to_agent_map[agent] in decision_steps_blue or not fixed_opponent:
                policy = policies[agent]
            elif fixed_opponent:
                policy = fixed_policy

            pool.apply_async(get_action, args=(policy, nn_input[agent], agent, q))

    pool.close()
    pool.join()
    print("JOINED")

    return q
