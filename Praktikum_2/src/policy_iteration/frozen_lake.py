import sys
import numpy as np
import gym


def evaluate_policy(_env, policy, _discount_rate, _epsilon):
    state_size = _env.observation_space.n

    # initialize value function
    V = np.zeros(state_size)

    while True:
        delta = 0

        for state in range(state_size):
            v = V[state]
            action = policy[state]
            # update value function using the current policy
            V[state] = sum([p * (r + _discount_rate * V[next_state]) for p, next_state, r, _ in _env.P[state][action]])
            delta = max(delta, abs(v - V[state]))

        if delta < _epsilon:
            break

    return V


def improve_policy(_env, V, _discount_rate):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    new_policy = np.zeros(state_size, dtype=int)

    for state in range(state_size):
        # find the best action for the current state
        action_values = [sum([p * (r + _discount_rate * V[next_state]) for p, next_state, r, _ in _env.P[state][action]])
                         for action in range(action_size)]
        new_policy[state] = np.argmax(action_values)

    return new_policy


def policy_iteration(_env, _num_iterations, _discount_rate, _epsilon):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    # initialize a random policy
    policy = np.random.choice(action_size, state_size)

    for _ in range(_num_iterations):
        # Policy Evaluation
        V = evaluate_policy(_env, policy, _discount_rate, _epsilon)

        # Policy Improvement
        new_policy = improve_policy(_env, V, _discount_rate)

        # Check for convergence
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy


def createOptimalPolicy(policy):
    optimalPolicy = []

    for field in policy:
        if field == 0:
            optimalPolicy.append("X")
        else:
            optimalPolicy.append(action_to_str(field))

    array_1d = np.array(optimalPolicy)
    array_2d = array_1d.reshape((4, 4))

    return array_2d


def action_to_str(action):
    return ["left", "down", "right", "up"][action]


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=True)

    # policy iteration hyperparameters
    num_iterations = 100
    discount_rate = 0.8
    epsilon = 1e-8

    # perform policy iteration
    optimal_values, optimal_policy = policy_iteration(env, num_iterations, discount_rate, epsilon)

    print("Optimal Values:")
    print(optimal_values.reshape(4, 4))

    print("Optimal Policy:")
    print(createOptimalPolicy(optimal_policy).reshape(4,4))
