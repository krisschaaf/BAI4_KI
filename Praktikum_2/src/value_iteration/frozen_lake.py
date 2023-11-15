import sys
import numpy as np
import gym


def value_iteration(env, num_iterations=10000, epsilon=1e-8, discount_rate=0.8):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # initialize value function
    optimal_policy = np.zeros(state_size)

    for _ in range(num_iterations):
        delta = 0

        for state in range(state_size):
            current_state = optimal_policy[state]
            action_values = []

            for action in range(action_size):
                state_value = 0
                for probability, next_state, reward, _ in env.P[state][action]:
                    state_action_value = probability * (reward + discount_rate * optimal_policy[next_state])
                    state_value += state_action_value
                action_values.append(state_value)

            optimal_policy[state] = max(action_values)

            delta = max(delta, abs(current_state - optimal_policy[state]))

        # delta converges to epsilon -> stop
        if delta < epsilon:
            break

    return optimal_policy


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=True)

    # value iteration hyperparameters
    num_iterations = 10000
    epsilon = 0.00000001  # converges to epsilon
    discount_rate = 0.8

    # perform value iteration
    optimal_policy = value_iteration(env, num_iterations, epsilon, discount_rate)

    print("Optimal Policy:")
    print(optimal_policy.reshape(4, 4))
