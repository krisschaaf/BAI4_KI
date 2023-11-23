import numpy as np
import gym


def value_iteration(_env, _num_iterations, _epsilon, _discount_rate):
    state_size = _env.observation_space.n
    action_size = _env.action_space.n

    # initialize value function
    V = np.zeros(state_size)

    for _ in range(_num_iterations):
        delta = 0

        for state in range(state_size):
            current_state = V[state]
            A = np.zeros(action_size)

            for action in range(action_size):
                for probability, next_state, reward, _ in _env.P[state][action]:
                    A[action] += probability * (reward + _discount_rate * V[next_state])

            V[state] = max(A)

            delta = max(delta, abs(current_state - V[state]))

        # delta converges to epsilon -> stop
        if delta < _epsilon:
            break

    policy = np.zeros([state_size, action_size])

    for state in range(state_size):  # for all states, create deterministic policy

        A = np.zeros(action_size)
        for action in range(action_size):
            for probability, next_state, reward, _ in _env.P[state][action]:
                A[action] += probability * (reward + _discount_rate * V[next_state])

        best_action = np.argmax(A)
        policy[state][best_action] = 1

    return V, policy


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
    env = gym.make('Taxi-v3', render_mode='ansi')

    # value iteration hyperparameters
    num_iterations = 10000
    epsilon = 0.00000001  # converges to epsilon
    discount_rate = 0.8

    # perform value iteration
    V, optimal_policy = value_iteration(env, num_iterations, epsilon, discount_rate)

    print("Optimal Values:")
    print(V)

    print("Optimal Policy:")
    print(optimal_policy)

    path = calculatePathByOptimalPolicy(0, 0, 0, 2, optimal_policy, env)
    print(path)