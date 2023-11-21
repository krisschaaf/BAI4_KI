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
            V[state] = sum([p * (r + _discount_rate * V[next_state]) for p, next_state, r, _ in _env.P[state][action]]) # TODO: reshape
            delta = max(delta, abs(v - V[state]))

        if delta < _epsilon:
            break

    return V


def improve_policy(_env, V, _discount_rate):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    new_policy = np.zeros(state_size, dtype=int)

    for state in range(state_size):
        action_values = []

        # Calculate action values for each action in the current state
        for action in range(action_size):
            transition_probabilities = _env.P[state][action]

            # Calculate the action value for the current action
            action_value = sum(
                [p * (r + _discount_rate * V[next_state]) for p, next_state, r, _ in transition_probabilities])

            # Append the action value to the list
            action_values.append(action_value)

        # Select the action with the maximum value for the current state
        new_policy[state] = np.argmax(action_values)

    return new_policy


def policy_iteration(_env, _num_iterations, _discount_rate, _epsilon):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    # initialize a random policy
    policy = np.random.choice(action_size, state_size)

    for _ in range(_num_iterations):
        # Policy Evaluation - return new value function
        V = evaluate_policy(_env, policy, _discount_rate, _epsilon)

        # Policy Improvement
        new_policy = improve_policy(_env, V, _discount_rate)

        # Check for convergence
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy


# def createOptimalPolicy(policy):
    # TODO
    # optimalPolicy = []
    #
    # for field in policy:
    #     optimalPolicy.append(action_to_str(field))
    #
    # array_1d = np.array(optimalPolicy)
    # array_2d = array_1d.reshape((4, 4))
    #
    # return array_2d


if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    # policy iteration hyperparameters
    num_iterations = 10000
    discount_rate = 0.9
    epsilon = 1e-8

    # perform policy iteration
    optimal_values, optimal_policy = policy_iteration(env, num_iterations, discount_rate, epsilon)

    print("Optimal Values:")
    print(optimal_values)

    print("Optimal Policy:")
    # print(createOptimalPolicy(optimal_policy))
    print(optimal_policy)
