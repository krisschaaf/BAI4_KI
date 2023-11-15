import gym
import numpy as np
from gym import *
import random


def main():
    env: gym.Env = gym.make('Taxi-v3')

    # initialize q-table
    state_size: Space = env.observation_space.n
    action_size: Space = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9  # alpha
    discount_rate = 0.8  # gamma, discount factor to give more or less importance to the next reward
    epsilon = 1.0        # explore vs exploit
    decay_rate = 0.001   # Fixed amount to decrease epsilon

    # training variables
    num_episodes = 10000
    counter_explore = 0
    counter_exploit = 0

    # training
    for episode in range(num_episodes):

        # reset the environment
        state, _ = env.reset()  # dont use info
        terminated = False
        truncated = False


        # for s in range(max_steps):
        while not terminated:

            # exploration vs exploitation
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
                counter_explore += 1
            else:
                # exploit
                action = np.argmax(qtable[state, :])
                counter_exploit += 1

            # take action and observe reward
            new_state, reward, terminated, truncated, _ = env.step(action)

            if truncated:
                print("Truncated: reset state")
                state, _ = env.reset()  # dont use info
                break

            # Q-learning algorithm
            qtable[state, action] = (
                    qtable[state, action] +   # (1-alpha) * Q(s,a) +
                    learning_rate *                                 # alpha * [ R(s,a,s’) + gamma * max’Q(s’,a’) ]
                    (
                            reward +
                            discount_rate * np.max(qtable[new_state, :]) -
                            qtable[state, action]
                    )
            )

            # Update to our new state
            state = new_state

        # Decrease epsilon
        epsilon = max(epsilon - decay_rate, 0)

    print(f"Training completed over {num_episodes} episodes")

    # watch trained agent
    env: gym.Env = gym.make('Taxi-v3', render_mode="rgb_array_list")
    state, _ = env.reset()
    rewards = 0

    for s in range(num_episodes):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state, :])
        new_state, reward, terminated, truncated, _ = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if terminated:
            if reward < 1.0:
                print("You're dead.")
                break
            else:
                print("You won - gg.")
                break

    print(f"Exploited: {counter_exploit}; Explored: {counter_explore}")
    # env.close()


if __name__ == "__main__":
    main()
