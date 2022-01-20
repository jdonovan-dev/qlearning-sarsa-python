import gym
import random
import numpy as np
import time
import pickle
from collections import deque
from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value)
    episode_reward_record = deque(maxlen=100)

    #TODO PERFORM SARSA LEARNING
    for i in range(EPISODES):
        episode_reward = 0.0
        obs = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)

            new_obs, reward, done, _ = env.step(action)

            if random.uniform(0, 1) < EPSILON:
                next_action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(new_obs,i)] for i in range(env.action_space.n)])
                next_action =  np.argmax(prediction)

            Q_table[obs, action] = Q_table[obs, action] + LEARNING_RATE * (reward +  DISCOUNT_FACTOR * np.max(np.array([Q_table[(new_obs, i)] for i in range(env.action_space.n)])) - Q_table[obs, action])
            obs = new_obs
            action = next_action
            episode_reward += reward

        EPSILON = EPSILON * EPSILON_DECAY
        episode_reward_record.append(reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )

    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################


