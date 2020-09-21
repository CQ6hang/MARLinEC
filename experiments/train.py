import random
import argparse
import numpy as np
from environment.env import Env
from marl.MARL_algorithm import DeepQNetwork
from marl.experiment_replay_pool import Memory


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--max-episodes", type=int, default=500, help="number of episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of data to optimize at the same time")
    parser.add_argument("--memory-size", type=int, default=10000, help="capacity of replay pool")
    parser.add_argument("--replace-target-iter", type=int, default=500, help="network parameter update interval")
    parser.add_argument("--e-greedy-decrement", type=float, default=2e-5, help="")
    parser.add_argument("--e-greedy", type=float, default=0.3, help="")
    parser.add_argument("--e-greedy-min", type=float, default=0.01, help="")
    return parser.parse_args()


def train(arglist):
    # create environment
    env = Env()

    memories = [Memory(arglist.memory_size) for _ in range(env.n_user)]

    dqn = [DeepQNetwork(len(env.action_space[i]), len(env.obs_space[i]), i,
                        learning_rate=arglist.lr,
                        reward_decay=arglist.gamma,
                        e_greedy=arglist.e_greedy,
                        e_greedy_min=arglist.e_greedy_min,
                        replace_target_iter=arglist.replace_target_iter,
                        memory_size=arglist.memory_size,
                        e_greedy_decrement=arglist.e_greedy_decrement
                        ) for i in range(env.n_user)]
    print('dqn build complete, start training...')
    for episode in range(arglist.max_episodes):
        step = 0
        rwd = [0.0 for _ in range(env.n_user)]
        a_rwd = [0.0 for _ in range(env.n_user)]
        obs = env.reset()

        while not all(env.done):
            # print(env.done)
            # print(step)
            step += 1
            actions = []
            for i in range(env.n_user):
                if np.random.uniform() < dqn[i].epsilon:
                    actions.append(np.random.randint(0, len(env.action_space[i])))
                else:
                    action = dqn[i].choose_action(obs[i])
                    actions.append(action)
            obs_, reward, done = env.step(actions)

            for i in range(env.n_user):
                if not env.done[i]:
                    rwd[i] += reward[i]
                    memories[i].remember(obs[i], actions[i], reward[i], obs_[i], done[i])
                    if step % 5 == 0:
                        size = memories[i].pointer
                        batch = random.sample(range(size), size) if size < arglist.batch_size else random.sample(
                            range(size), arglist.batch_size)
                        dqn[i].learn(*memories[i].sample(batch))
                else:
                    a_rwd[i] = rwd[i] / step

            obs = obs_

        if episode % 10 == 0:
            print(
                'episode:' + str(episode) + ' steps:' + str(step) +
                '\nreward:' + str(rwd) + '\naverage_reward:' + str(a_rwd))




if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
