print('\nimporting\n')
from typing import Optional

import collections
from dm_control import suite as dm_suite
import dm_env
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

from acme import specs
from acme import wrappers
from acme.agents.jax import d4pg
from acme.agents.jax import r2d2
from acme.jax import experiments
from acme.utils import loggers

ENV_NAME = "Pong"

from typing import Tuple
import gym
import functools
import haiku as hk
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils

print('\nmaking atari env func\n')

def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    oar_wrapper: bool = False,
    num_stacked_frames: int = 4,
    flatten_frame_stack: bool = False,
    grayscaling: bool = True,
    to_float: bool = True,
    scale_dims: Tuple[int, int] = (84, 84),
) -> dm_env.Environment:
    """Loads the Atari environment."""
    # Internal logic.
    version = 'v0' if sticky_actions else 'v4'
    level_name = f'{level}NoFrameskip-{version}'
    env = gym.make(level_name, full_action_space=True)

    wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=108_000,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            zero_discount_on_life_loss=zero_discount_on_life_loss,
        ),
        wrappers.SinglePrecisionWrapper,
        ]

    if oar_wrapper:
    # E.g. IMPALA and R2D2 use this particular variant.
        wrapper_list.append(wrappers.ObservationActionRewardWrapper)

    return wrappers.wrap_all(env, wrapper_list)

print('\nmaking dqn network func\n')


def make_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
    """Creates networks for training DQN on Atari."""
    def network(inputs):
        model = hk.Sequential([
            networks_lib.AtariTorso(),
            hk.nets.MLP([512, environment_spec.actions.num_values]),
        ])
        return model(inputs)
    network_hk = hk.without_apply_rng(hk.transform(network))
    obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
    typed_network = networks_lib.non_stochastic_network_to_typed(network)
    return dqn.DQNNetworks(policy_network=typed_network)


print('\nmaking env func\n')

def make_environment(seed: int) -> dm_env.Environment:
  # def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return make_atari_environment(
        level=ENV_NAME,
        sticky_actions=True,
        zero_discount_on_life_loss=False,
        oar_wrapper=True,
        num_stacked_frames=1,
        flatten_frame_stack=True,
        grayscaling=False)  # environment = dm_suite.load('cartpole', 'balance')

  # # Make the observations be a flat vector of all concatenated features.
  # environment = wrappers.ConcatObservationWrapper(environment)

  # # Wrap the environment so the expected continuous action spec is [-1, 1].
  # # Note: this is a no-op on 'control' tasks.
  # environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

  # # Make sure the environment outputs single-precision floats.
  # environment = wrappers.SinglePrecisionWrapper(environment)
  # print('\n\nenv made\n\n')
  # return environment

print('\nmaking network factory func\n')

def network_factory(spec: specs.EnvironmentSpec) -> r2d2.R2D2Networks:
    return r2d2.make_atari_networks(
        spec,
        # These correspond to sizes of the hidden layers of an MLP.
        # policy_layer_sizes=(256, 256),
        # critic_layer_sizes=(256, 256),
    )

print('\nmaking config and builder\n')

r2d2_config = r2d2.R2D2Config(learning_rate=3e-4)
r2d2_builder = r2d2.R2D2Builder(r2d2_config)

# Specify how to log training data: in this case keeping it in memory.
# NOTE: We create a dict to hold the loggers so we can access the data after
# the experiment has run.

print('\nmaking logger dict and func\n')
logger_dict = collections.defaultdict(loggers.InMemoryLogger)
def logger_factory(
    name: str,
    steps_key: Optional[str] = None,
    task_id: Optional[int] = None,
    ) -> loggers.Logger:
    del steps_key, task_id
    return logger_dict[name]

print('\nmaking experiment config\n')

experiment_config = experiments.ExperimentConfig(
    builder=r2d2_builder,
    environment_factory=make_environment,
    network_factory=network_factory,
    logger_factory=logger_factory,
    seed=0,
    max_num_actor_steps=50_000)  # Each episode is 1000 steps.

print('\nrunning experiment\n')
experiments.run_experiment(
    experiment=experiment_config,
    eval_every=1000,
    num_eval_episodes=1)

print('\nsuccess! running\n')
df = pd.DataFrame(logger_dict['evaluator'].data)
plt.figure(figsize=(10, 4))
plt.title('Training episodes returns')
plt.xlabel('Training episodes')
plt.ylabel('Episode return')
plt.plot(df['actor_episodes'], df['episode_return'], label='Training Episodes return')
plt.savefig('results.png')