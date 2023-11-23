import sys
import gym
import numpy as np
from gym import *
import random
from IPython.display import clear_output


def q_learning(_env, _qtable, _num_iterations, _epsilon, _discount_rate, _learning_rate, _decay_rate):
    counter_explore = 0
    counter_exploit = 0

    # training
    for _ in range(_num_iterations):
        delta = 0

        # reset the environment
        state, _ = _env.reset()  # dont use info
        terminated = False

        while not terminated:

            # exploration vs exploitation
            if random.uniform(0, 1) < _epsilon:
                # explore
                action = _env.action_space.sample()
                counter_explore += 1
            else:
                # exploit
                action = np.argmax(_qtable[state, :])
                counter_exploit += 1

            # take action and observe reward
            new_state, reward, terminated, truncated, _ = _env.step(action)

            if truncated:
                print("Truncated: reset state")
                state, _ = _env.reset()  # dont use info
                break

            # Q-learning algorithm
            _qtable[state, action] = (
                    _qtable[state, action] +  # (1-alpha) * Q(s,a) +
                    _learning_rate *  # alpha * [ R(s,a,s’) + gamma * max’Q(s’,a’) ]
                    (
                            reward +
                            _discount_rate * np.max(_qtable[new_state, :]) -
                            _qtable[state, action]
                    )
            )

            # Update to our new state
            state = new_state

        # Decrease epsilon
        _epsilon = max(_epsilon - _decay_rate, 0)

    print(f"Training completed over {_num_iterations} episodes")
    print(f"\r\nExploited: {counter_exploit}; Explored: {counter_explore}")


def watchTrainedAgent(_num_iterations, _qtable):
    # watch trained agent
    _env: gym.Env = gym.make('Taxi-v3')
    state, _ = _env.reset()
    rewards = 0

    for s in range(_num_iterations):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(_qtable[state, :])
        new_state, reward, terminated, truncated, _ = _env.step(action)
        rewards += reward
        print(f"score: {rewards}")
        state = new_state

        if terminated:
            if reward < 1.0:
                print("You're dead.")
                break
            else:
                print("You won - gg.")
                break

    np.set_printoptions(threshold=sys.maxsize)


def calculatePathByOptimalPolicy(taxi_row, taxi_column, passenger_location, destination, _optimal_policy, _env):
    passenger_matrix = np.array([[0, 0], [0, 4], [4, 0], [4, 3]])
    pathByOptimalPolicy = []
    _env.reset()

    while True:
        state = _env.encode(taxi_row, taxi_column, passenger_location, destination)
        action = np.argmax(_optimal_policy[state])
        pathByOptimalPolicy.append(state)
        _env.s = state
        p = _env.render()
        print(p)
        match action:
            case 0:
                taxi_column += 1
            case 1:
                taxi_column -= 1
            case 2:
                taxi_row += 1
            case 3:
                taxi_row -= 1
            case 4:
                if passenger_location == 4:
                    raise Exception("Taxi trying to pick up passenger who is already inside the car.")

                if np.array_equal(passenger_matrix[passenger_location], [taxi_row, taxi_column]):
                    passenger_location = 4
                else:
                    raise Exception("Taxi trying to pick up passenger who is not on the designated pick up field.")
            case 5:
                if passenger_location != 4:
                    raise Exception("Taxi trying to drop off passenger who is not inside the car.")

                if not np.array_equal([taxi_row, taxi_column], passenger_matrix[destination]):
                    state = _env.encode(taxi_row, taxi_column, passenger_location, destination)
                    pathByOptimalPolicy.append(state)

                    break
                else:
                    raise Exception("Taxi trying to drop off passenger at the wrong destination.")

    return pathByOptimalPolicy


if __name__ == "__main__":
    env: gym.Env = gym.make('Taxi-v3', render_mode="ansi")
    # hyperparameters
    learning_rate = 0.92  # alpha
    discount_rate = 0.95  # gamma, discount factor to give more or less importance to the next reward
    epsilon = 0.9  # explore vs exploit
    decay_rate = 0.005  # Fixed amount to decrease epsilon
    num_iterations = 5000

    state_size: Space = env.observation_space.n
    action_size: Space = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    q_learning(env, qtable, num_iterations, discount_rate, epsilon, learning_rate, decay_rate)
    watchTrainedAgent(num_iterations, qtable)


    print("\r\nOptimal Policy:")
    print(qtable)

    path = calculatePathByOptimalPolicy(0, 0, 0, 2, qtable, env)
    print(path)
