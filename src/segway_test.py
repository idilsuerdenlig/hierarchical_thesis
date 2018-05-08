import numpy as np

from mushroom.algorithms.actor_critic import SAC_AVG
from mushroom.core import Core
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.policy import StateLogStdGaussianPolicy, StateStdGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import Parameter
from library.environments.segway import Segway

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_episodes):
    np.random.seed()

    # MDP
    mdp = Segway()

    # Agent
    n_tilings = 11
    alpha_r = Parameter(.0001)
    alpha_theta = Parameter(0.001/n_tilings)
    alpha_v = Parameter(.1/n_tilings)

    tilings_v = Tiles.generate(10, [5, 5, 5],
                               mdp.info.observation_space.low,
                               mdp.info.observation_space.high+1e-3)
    psi = Features(tilings=tilings_v)

    mu = Regressor(LinearApproximator,
                   input_shape=mdp.info.observation_space.shape,
                   output_shape=mdp.info.action_space.shape)

    std = Regressor(LinearApproximator,
                    input_shape=mdp.info.observation_space.shape,
                    output_shape=mdp.info.action_space.shape)

    std_0 = np.sqrt(1.0)
    std.set_weights(np.log(std_0)/n_tilings*np.ones(std.weights_size))

    policy = StateLogStdGaussianPolicy(mu, std)

    agent = SAC_AVG(policy, mdp.info,
                    alpha_theta, alpha_v, alpha_r,
                    lambda_par=0.5,
                    value_function_features=psi)

    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_steps_per_fit=1, render=False)
        J = compute_J(dataset_callback.get(), gamma=1.0)
        dataset_callback.clean()
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.sum(J)/n_episodes))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_steps=5000, render=True)

if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 5

    experiment(n_epochs, n_episodes)
