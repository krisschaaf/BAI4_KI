import gym
import numpy as np
from gym import *
import random


def main():
    # env: gym.Env = gym.make('Taxi-v3', disable_env_checker=True)
    env: gym.Env = gym.make('FrozenLake-v1', is_slippery=False)

    # initialize q-table
    state_size: Space = env.observation_space.n
    action_size: Space = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9  # alpha
    discount_rate = 0.8  # gamma, discount factor to give more or less importance to the next reward
    epsilon = 1.0  # explore vs exploit

    # training variables
    num_episodes = 10000
    max_steps = 100  # per episode
    counter_explore = 0
    counter_exploit = 0

    # training
    for episode in range(num_episodes):

        # reset the environment
        state, _ = env.reset()  # dont use info
        terminated = False
        truncated = False

        action = env.action_space.sample()

        for s in range(max_steps):
        # while not terminated:

            # exploration vs exploitation
            if random.uniform(0, 1) < epsilon:
                # explore
                new_action = env.action_space.sample()
                while truncated and new_action == action:
                    new_action = env.action_space.sample()

                action = new_action

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
                    (1 - learning_rate) * qtable[state, action] +   # (1-alpha) * Q(s,a) +
                    learning_rate *                                 # alpha * [ R(s,a,s’) + gamma * max’Q(s’,a’) ]
                    (
                            reward +
                            discount_rate * np.max(qtable[new_state, :])
                    )
            )

            # Update to our new state
            state = new_state

            # # if terminated, finish episode
            if terminated:
                break

        # Decrease epsilon
        epsilon = np.exp(-0.005 * episode)

    print(f"Training completed over {num_episodes} episodes")

    # watch trained agent
    env: gym.Env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    state, _ = env.reset()
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state, :])
        new_state, reward, terminated, truncated, _ = env.step(action)
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

    print(f"Exploited: {counter_exploit}; Explored: {counter_explore}")
    # env.close()


if __name__ == "__main__":
    main()
