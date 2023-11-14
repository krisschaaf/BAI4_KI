import numpy as np
import gym
import random


def main():
    # create Taxi environment
    env: gym.Env = gym.make('Taxi-v3')

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99  # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe reward
            new_state, reward, done, info, _ = env.step(action)

            bar = discount_rate * np.max(qtable[new_state, :])

            baz = qtable[state, action]

            foo = (reward + bar - baz)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * foo

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, done, info, _ = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
