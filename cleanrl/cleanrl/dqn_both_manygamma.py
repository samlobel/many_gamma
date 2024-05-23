# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
import pickle
from collections import defaultdict
from enum import Enum

from gamma_utilities import *
from gradient_based_coefficients import CoefficientsModule
import tabular_environments # for gymnasium registration
from q_networks import ManyGammaQNetwork, get_upper_and_lower_bound_pairwise_constraints

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import custom_envs # ZeroRewardEnv-v0, OneRewardEnv-v0


class InitializationModes(Enum):
    RANDOM = 0
    OPTIMAL = 1
    OPTIMAL_LAST_GAMMA_ZERO = 2
    OPTIMAL_LAST_GAMMA_DOUBLE = 3
    OPTIMAL_FIRST_GAMMA_ZERO = 4
    OPTIMAL_FIRST_GAMMA_DOUBLE = 5
    OPTIMAL_PLUS_NOISE = 6
    OPTIMAL_PLUS_LOTS_OF_NOISE = 7

@dataclass
class ArgsBase:
    # I don't know why I didn't have a base class before, should make it clearer what changes.
    # Started as tabular.
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1" # DIFFERS
    """the id of the environment"""
    total_timesteps: int = 500000 # DIFFERS
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4 # DIFFERS
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000 # DIFFERS
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500 # DIFFERS
    """the timesteps it takes to update the target network"""
    batch_size: int = 128 # DIFFERS
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05 # DIFFERS
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5 # DIFFERS
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000 # DIFFERS
    """timestep to start learning"""
    train_frequency: int = 10 # DIFFERS
    """the frequency of training"""
    constraint_loss_scale: float = 0.0
    """the scale of the constraint loss. Set to 0 (default) to disable constraint loss"""
    pairwise_loss_scale: float = 0.0
    """the scale of the constraint loss. Set to 0 (default) to disable constraint loss"""
    # all_gammas: tuple[float] = (0.98, 0.99)
    # all_gammas: str = "0.98,0.99"
    gamma_lower: float = 0.98
    """Smallest of the evenly-spaced Gammas"""
    gamma_upper: float = 0.99
    """Largest of the evenly-spaced Gammas"""
    num_gammas: int = 2
    """How many gammas to train"""
    tag: str = ""
    """Directory name for experiment"""
    log_dir: str = "runs"
    """One above directory name for experiment"""
    is_atari: bool = False
    """Determines Network Shape etc"""
    is_minatar: bool = False
    """Determines Network Shape etc"""
    is_tabular: bool = False
    """Determines stuff like logging I think"""
    semigradient_constraint: bool = False
    """If true, detaches constraints in constraint loss"""
    constraint_regularization: float = 0.0
    """Nonzero values smooth out coefficients"""
    constraint_normalization: str = None
    """None, 'l1', or 'l2'. Whether we normalize Q values before constraint lossing."""
    coefficient_metric: str = "l2"
    """Either `l2` or `abs`, determines whether we do the analytic solution or gradient based solution."""
    gamma_spacing: str = "even"
    """Even, log, or linear."""
    only_from_lower: bool = False
    """Whether to only constrain from lower gammas"""
    skip_self_map: bool = False
    """Whether or not to use the current gamma for constraining"""
    r_min: float = 0.0
    """Minimum per-step reward"""
    r_max: float = 1.0
    """Maximum per-step reward"""
    cap_with_vmax: bool = False
    """Whether to cap values with 1/(1-gamma) before inputting to constraint matrix, also keeps constraints below the same."""
    pairwise_constraint: bool = False
    """Make a constrain using the log exp math from the writeup"""
    scale_constraint_loss_by_vmax: bool = False
    """Whether to scale each element of constraint loss by 1/(1-gamma)"""
    additive_constant: float = 0.
    """Whether to scale each element of constraint loss by 1/(1-gamma)"""
    additive_multiple_of_vmax: float = 0.
    """Whether to scale each element of constraint loss by 1/(1-gamma)"""
    neural_net_multiplier: float = 1.0
    """Something that gives us a simple knob to increase the output scale of the learned component."""
    vmax_cap_method: str = "pre-coefficient" # pre-coefficient, post-coefficient, separate-regularization
    """How to cap the Q-values. pre-coefficient, post-coefficient, separate-regularization"""
    optimizer: str = "adam"
    """Which optimizer to use. Choices are adam and sgd"""
    td_loss_scale: float = 1.0
    """How much to weight TD loss by. Set to 0 for only constraint optimization."""
    # initialize_to_optimal: bool = False
    # """Whether to initialize the Q-values to the optimal values (if tabular)."""
    tabular_initialization_mode: int = InitializationModes.RANDOM.value
    # These are up here because we can't add new things elsewhere, because we parse using argsclassic first. Silly but don't want to refactor.
    tabular_kwargs_num_states: int = None
    """Number of states in the tabular environment"""
    tabular_kwargs_num_actions: int = None
    """Number of actions in the tabular environment"""
    tabular_kwargs_amount_noise_prob: float = None
    """Amount of noise in the tabular environment"""
    apply_constraint_to_target: bool = False
    """Whether to clip and propagate target values"""
    use_clipping_for_target: bool = False
    """Whether to clip and propagate target values"""
    double_q_learning: bool = False
    """Whether to use q_network for target_network's action selection"""



@dataclass
class ArgsTabular(ArgsBase):
    env_id: str = "RandomTabularEnv-v0"
    """the id of the environment"""
    learning_rate: float = 1e-2
    """the learning rate of the optimizer"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    train_frequency: int = 1
    """the frequency of training"""
    is_tabular: bool = True
    """Determines stuff like logging I think"""

@dataclass
class ArgsClassic(ArgsBase):
    # Since just same as before.
    pass


@dataclass
class ArgsAtari(ArgsBase):
    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

@dataclass
class ArgsMinAtar(ArgsBase):
    env_id = "MinAtar/Breakout-v1"
    """the id of the environment"""
    is_minatar: bool = True
    """Whether MinAtar"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4 # There are also other optimizer parameters: grad_momentum, squared_grad_momentum.
    """the learning rate of the optimizer"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    train_frequency: int = 1
    """the frequency of training"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    exploration_fraction: float = 0.02
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 5000
    """timestep to start learning"""


class MinAtarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_shape = env.observation_space.shape
        assert len(observation_shape) == 3
        assert observation_shape[0] == observation_shape[1] == 10
        new_observation_shape = (observation_shape[2], observation_shape[0], observation_shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_observation_shape, dtype=np.float32)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def make_env(env_id, seed, idx, capture_video, run_name, args):
    def thunk_atari():
        # TODO: Do I want this part changed to use something more like run_dir? Not sure how that will effect things like w&b
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env
    
    def thunk_flat():
        # TODO: Do I want this part changed to use something more like run_dir? Not sure how that will effect things like w&b
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    def thunk_minatar():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MinAtarWrapper(env)
        env.action_space.seed(seed)
        return env

    def thunk_tabular():
        tabular_kwargs = {}
        for key in ['num_states', 'num_actions', 'amount_noise_prob']:
            arg_key = "tabular_kwargs_" + key
            arg_val = getattr(args, arg_key, None)
            if arg_val is not None:
                tabular_kwargs[key] = arg_val
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **tabular_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **tabular_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    if args.is_atari:
        return thunk_atari
    if args.is_minatar:
        return thunk_minatar
    if args.is_tabular:
        return thunk_tabular
    return thunk_flat


# ALGO LOGIC: initialize agent here:




def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(ArgsClassic)

    num_is_flags = sum([1 if val else 0 for val in [args.is_atari, args.is_tabular, args.is_minatar]])
    assert num_is_flags <= 1, "Shouldn't be more than one"

    if args.is_atari:
        args = tyro.cli(ArgsAtari) # Different defaults
    if args.is_tabular:
        args = tyro.cli(ArgsTabular) # Different defaults
    if args.is_minatar:
        args = tyro.cli(ArgsMinAtar)

    if args.is_minatar:
        args.env_id in ("Asterix-v1", "Breakout-v1", "Freeway-v1", "Seaquest-v1", "SpaceInvaders-v1",)
        args.env_id = f"MinAtar/{args.env_id}"
        assert args.env_id.startswith("MinAtar/")

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    # if not args.cap_with_vmax: # Actually I won't do this as to not mess up my onager stuff.
    #     assert args.vmax_cap_method == "pre-coefficient", "Cause its the default" 
    assert args.vmax_cap_method in ("pre-coefficient", "post-coefficient", "separate-regularization"), args.vmax_cap_method
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.tag:
        run_name = args.tag
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.tag + '__' if args.tag else ''}{int(time.time())}"

    if args.tabular_initialization_mode != InitializationModes.RANDOM.value:
        assert args.is_tabular, "Can only initialize to optimal if tabular"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    write_dir = os.path.join(args.log_dir, run_name)
    # writer = SummaryWriter(f"runs/{run_name}")
    writer = SummaryWriter(write_dir)
    pkl_log_path = os.path.join(write_dir, "log_dict.pkl")
    log_dict = defaultdict(list)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args=args) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    assert args.gamma_spacing in ("even", "log", "linear"), args.gamma_spacing
    if args.num_gammas == 1:
        gammas = [args.gamma_upper]
    else:
        gamma_choosing_func = {"even": get_even_spacing, "log": get_even_log_spacing, "linear": np.linspace}[args.gamma_spacing]
        gammas = gamma_choosing_func(args.gamma_lower, args.gamma_upper, args.num_gammas)
    print(gammas)

    if args.is_tabular:
        single_env = envs.envs[0]
        true_q_values = single_env.get_manygamma_values(gammas, gamma_to_choose=gammas[-1])
        # true_q_values = []
        # for g in gammas:
        #     # print(g)
        #     true_q_values.append(single_env.get_optimal_q_values_and_policy(g)[0][:,None,:])
        # # true_q_values = [env.get_optimal_q_values_and_policy(g)[0][:,None,:] for g in gammas] # Add gamma dimension to stack
        # true_q_values = np.concatenate(true_q_values, axis=1) # [num_states, num_gammas, num_actions]
        # print('neato')

    q_initialization = None
    if args.is_tabular:
        optimal_tensor = torch.tensor(true_q_values)
        if args.tabular_initialization_mode == InitializationModes.RANDOM.value:
            q_initialization = None
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL.value:
            q_initialization = optimal_tensor.clone()
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_LAST_GAMMA_ZERO.value:
            q_initialization = optimal_tensor.clone()
            q_initialization[:, -1, :] = 0.
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_LAST_GAMMA_DOUBLE.value:
            q_initialization = optimal_tensor.clone()
            q_initialization[:, -1, :] *= 2.
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_FIRST_GAMMA_ZERO.value:
            q_initialization = optimal_tensor.clone()
            q_initialization[:, 0, :] = 0.
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_FIRST_GAMMA_DOUBLE.value:
            q_initialization = optimal_tensor.clone()
            q_initialization[:, 0, :] *= 2.
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_PLUS_NOISE.value:
            q_initialization = optimal_tensor.clone()
            q_initialization += (2 * torch.rand(*q_initialization.shape) - 1)
        elif args.tabular_initialization_mode == InitializationModes.OPTIMAL_PLUS_LOTS_OF_NOISE.value:
            q_initialization = optimal_tensor.clone()
            q_initialization += (20 * torch.rand(*q_initialization.shape) - 10)
        else:
            raise Exception("Should have caught by now")

    # So, it doesn't transfer coefficients etc. This is actually bad if we use the target network's constraints
    q_network = ManyGammaQNetwork(
        envs, gammas, constraint_regularization=args.constraint_regularization,
        metric=args.coefficient_metric, only_from_lower=args.only_from_lower, skip_self_map=args.skip_self_map,
        r_min=args.r_min, r_max=args.r_max,
        cap_with_vmax=args.cap_with_vmax,
        vmax_cap_method=args.vmax_cap_method,
        additive_constant=args.additive_constant,
        additive_multiple_of_vmax=args.additive_multiple_of_vmax,
        neural_net_multiplier=args.neural_net_multiplier,
        is_tabular=args.is_tabular,
        is_atari=args.is_atari,
        is_minatar=args.is_minatar,
        initialization_values=q_initialization,
        # initialize_to_optimal=args.initialize_to_optimal,
        # optimal_init_values=torch.tensor(true_q_values) if args.initialize_to_optimal else None,
        ).to(device)
    assert args.optimizer.lower() in ("adam", "sgd"), args.optimizer
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(q_network.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(q_network.parameters(), lr=args.learning_rate)
    target_network = ManyGammaQNetwork(
        envs, gammas, constraint_regularization=args.constraint_regularization,
        metric=args.coefficient_metric, only_from_lower=args.only_from_lower, skip_self_map=args.skip_self_map,
        r_min=args.r_min, r_max=args.r_max,
        cap_with_vmax=args.cap_with_vmax,
        vmax_cap_method=args.vmax_cap_method,
        additive_constant=args.additive_constant,
        additive_multiple_of_vmax=args.additive_multiple_of_vmax,
        neural_net_multiplier=args.neural_net_multiplier,
        is_tabular=args.is_tabular,
        is_atari=args.is_atari,
        is_minatar=args.is_minatar,
        initialization_values=q_initialization,
        # initialize_to_optimal=args.initialize_to_optimal,
        # optimal_init_values=torch.tensor(true_q_values) if args.initialize_to_optimal else None,
        ).to(device)
    target_network.load_state_dict(q_network.state_dict())


    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        # optimize_memory_usage=True,
        optimize_memory_usage=False, # When true it doesn't handle termination correctly.
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # q_values = q_network(torch.Tensor(obs).to(device))
            # actions = torch.argmax(q_values, dim=1).cpu().numpy()
            actions = q_network.get_best_actions_and_values(torch.Tensor(obs).to(device))[0].cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    log_dict['episodic_return'].append((global_step, info["episode"]["r"].item()))
                    log_dict['episodic_length'].append((global_step, info["episode"]["l"].item()))
                    pkl_write_start = time.time()
                    with open(pkl_log_path, 'wb') as f:
                        pickle.dump(log_dict, f)
                    print(f"pkl write time: {time.time() - pkl_write_start:.4f}s")

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # 
        # if args.is_tabular:
        #     max_obs_index = np.argmax(obs[0])
        #     max_next_obs_index = np.argmax(real_next_obs[0])
        #     assert np.absolute((max_obs_index - max_next_obs_index)%10) == 1 or np.absolute((max_next_obs_index - max_obs_index)%10) == 1, f"{max_obs_index} to {max_next_obs_index}"

        rb.add(obs.copy(), real_next_obs.copy(), actions.copy(), rewards.copy(), terminations.copy(), infos.copy())

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        # TODO: add DDQN feature, in case helpful.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # 
                # if args.is_tabular:
                #     assert data.dones.max() == data.dones.min() == 0.
                #     assert data.observations.sum(dim=1).min() == data.observations.sum(dim=1).max() == 1.
                #     assert data.next_observations.sum(dim=1).min() == data.next_observations.sum(dim=1).max() == 1.
                # max_obs_index_batch = np.argmax(data.observations.cpu().numpy(), axis=1)
                # max_next_obs_index_batch = np.argmax(data.next_observations.cpu().numpy(), axis=1)
                # for i in range(len(max_obs_index_batch)):
                #     in_seen_set = (tuple(data.observations[i].tolist()), tuple(data.next_observations[i].tolist())) in s_sp_set
                #     assert in_seen_set, f"{(tuple(data.observations[i].tolist()), tuple(data.next_observations[i].tolist()))} not in seen"
                #     if args.is_tabular:
                #         right = (np.absolute((max_obs_index_batch[i] - max_next_obs_index_batch[i])%10) == 1 or np.absolute((max_next_obs_index_batch[i] - max_obs_index_batch[i])%10) == 1)
                #         # if not right:
                #         #     print('pause')
                #         #     print('resume')
                #         # print('bang')
                #         assert right, f"Batch[{i}]: {max_obs_index_batch[i]} to {max_next_obs_index_batch[i]}"
                #         assert np.absolute((max_obs_index_batch[i] - max_next_obs_index_batch[i])%10) == 1 or np.absolute((max_next_obs_index_batch[i] - max_obs_index_batch[i])%10) == 1, f"Batch[{i}]: {max_obs_index_batch[i]} to {max_next_obs_index_batch[i]}"
                with torch.no_grad():
                    # Not max for all gammas, but action is max action for main gamma.
                    # target_max_all_gammas = q_network.get_best_actions_and_values(torch.Tensor(obs).to(device))[1].cpu().numpy()
                    # target_max, _ = target_network(data.next_observations).max(dim=1)
                    # td_target_all_gammas = data.rewards.flatten() + args.gamma * target_max_all_gammas * (1 - data.dones.flatten())
                    q_for_action_selection = q_network if args.double_q_learning else None
                    td_target = target_network.get_target_value(data.next_observations, data.rewards, data.dones,
                                                                pass_through_constraint=args.apply_constraint_to_target, cap_by_vmax=args.use_clipping_for_target,
                                                                q_for_action_selection=q_for_action_selection)

                    # without_clipping = target_network.get_target_value(data.next_observations, data.rewards, data.dones,
                    #                                                     pass_through_constraint=False, cap_by_vmax=False)
                    # difference_from_clipping = td_target - without_clipping
                    # how_much_clipping_changes = difference_from_clipping[:, 0].mean()
                    # # print(f"clipping changes smallest gamma's val by {how_much_clipping_changes:.4f}")
                    # # print(f"Smallest gamma value now {td_target[:,0].mean():.4f}")
                    # if global_step % 1000 == 0:
                    #     without_clipping = target_network.get_target_value(data.next_observations, data.rewards, data.dones,
                    #                                                        pass_through_constraint=False, cap_by_vmax=False)
                    #     difference_from_clipping = without_clipping - td_target
                        # print('pausing')
                        # import ipdb; ipdb.set_trace()
                        # print('resuming')

                    # if args.apply_constraint_to_target:
                    #     use_clipping_for_target = args.use_clipping_for_target
                    #     td_target = q_network.propagate_and_bound_v(td_target, use_clipping_for_target)
                    # raise Exception("Here")
                    # td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                # old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # Cache it!
                q_outputs = q_network(data.observations)
                # old_val = q_network.get_values_for_action(data.observations, data.actions.squeeze())
                old_val = q_network.get_values_for_action(data.observations, data.actions.squeeze(), output=q_outputs) # [bs, num_gammas]
                # print(old_val[:,0].mean().item(), td_target[:,0].mean().item())
                # print(old_val[:,0].mean().item() - td_target[:,0].mean().item())
                # print((old_val[:,0] - td_target[:,0]).min().item(), (old_val[:,0] - td_target[:,0]).max().item())
                # raise Exception("here")
                td_loss = F.mse_loss(td_target, old_val)
                last_gamma_td_loss = F.mse_loss(td_target[:, -1], old_val[:, -1]) # was bs x n_gammas (no actions anymore)
                first_gamma_td_loss = F.mse_loss(td_target[:, 0], old_val[:, 0]) # was bs x n_gammas (no actions anymore)

                violation_dict = q_network.get_constraint_computed_values_and_violations(
                    data.observations, semi_gradient=args.semigradient_constraint,
                    normalize=args.constraint_normalization, output=q_outputs)
                upper_violations = violation_dict['upper_violations'] # actions last
                lower_violations = violation_dict['lower_violations'] # actions last
                # cap violations is batch, gamma, actions
                cap_violations_average = 0.5 * (violation_dict['cap_violations'] ** 2).mean()
                last_gamma_cap_violations_average = (violation_dict['cap_violations'][:, -1, :] ** 2).mean()
                last_gamma_constraint_loss = 0.5 * ((upper_violations[:, -1, :]**2).mean() + (lower_violations[:, -1, :]**2).mean())
                # td_loss = loss
                # constraint_loss = (upper_violations.mean() + lower_violations.mean())
                constraint_loss = 0.5*((upper_violations**2).mean() + (lower_violations**2).mean())

                scaled_upper_violations = upper_violations * (1 - q_network._gammas[None, :, None])
                scaled_lower_violations = lower_violations * (1 - q_network._gammas[None, :, None])
                scaled_cap_violations_average = 0.5*((violation_dict['cap_violations'] * (1 - q_network._gammas[None, :, None]))** 2).mean()
                scaled_constraint_loss = 0.5*((scaled_upper_violations**2).mean() + (scaled_lower_violations**2).mean())
                # scaled_cap_violaion_average =

                if args.constraint_loss_scale > 0:
                    if args.scale_constraint_loss_by_vmax:
                        total_loss = (args.td_loss_scale * td_loss) + (args.constraint_loss_scale * scaled_constraint_loss)
                        if args.cap_with_vmax and args.vmax_cap_method == "separate-regularization": # Don't love how messy this is.
                            total_loss += args.constraint_loss_scale * scaled_cap_violations_average
                    else:
                        total_loss = (args.td_loss_scale * td_loss) + (args.constraint_loss_scale * constraint_loss)
                        if args.cap_with_vmax and args.vmax_cap_method == "separate-regularization":
                            total_loss += args.constraint_loss_scale * cap_violations_average
                else:
                    total_loss = (args.td_loss_scale * td_loss)

                upper_pairwise = violation_dict['upper_pairwise'].detach()
                lower_pairwise = violation_dict['lower_pairwise'].detach()
                # Very frustrating I wrote it this way. If only I had the authority to change it.
                q_outputs_repeated = q_outputs.unsqueeze(1).repeat(1, q_outputs.shape[1], 1, 1)
                pairwise_violations_above = torch.maximum(q_outputs_repeated - upper_pairwise, torch.tensor(0, dtype=torch.float32))
                pairwise_violations_below = torch.maximum(lower_pairwise - q_outputs_repeated, torch.tensor(0, dtype=torch.float32))
                pairwise_violations_recomputed = torch.maximum(pairwise_violations_above, pairwise_violations_below)

                # pairwise_violations_recomputed = torch.maximum(q_outputs_repeated - upper_pairwise, lower_pairwise - q_outputs_repeated)
                pairwise_violation_mse = (pairwise_violations_recomputed ** 2).mean()

                if args.pairwise_loss_scale > 0:
                    # Seems like I compute losses in this file, annoyingly. So I should do it from the upper/lower bounds.
                    total_loss = total_loss + args.pairwise_loss_scale * pairwise_violation_mse



                log_frequency = 1000 if (args.is_atari or args.is_minatar) else 100
                if global_step % log_frequency == 0:
                    # writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/td_loss", td_loss, global_step)
                    writer.add_scalar("losses/total_loss", total_loss, global_step)
                    # if args.constraint_loss_scale > 0:
                    writer.add_scalar("losses/constraint_loss", constraint_loss, global_step)
                    writer.add_scalar("losses/constraint_loss", scaled_constraint_loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    log_dict['td_loss'].append((global_step, td_loss.item()))
                    log_dict['total_loss'].append((global_step, total_loss.item()))
                    log_dict['constraint_loss'].append((global_step, constraint_loss.item()))
                    log_dict['scaled_constraint_loss'].append((global_step, scaled_constraint_loss.item()))
                    log_dict['pairwise_violation_mse'].append((global_step, pairwise_violation_mse.item()))
                    log_dict['q_values'].append((global_step, old_val.mean().item()))
                    log_dict['SPS'].append((global_step, int(global_step / (time.time() - start_time))))
                    log_dict['cap_violations_average'].append((global_step, cap_violations_average.item()))
                    log_dict['scaled_cap_violations_average'].append((global_step, scaled_cap_violations_average.item()))

                    log_dict['last_gamma_cap_violations_average'].append((global_step, last_gamma_cap_violations_average.item()))
                    log_dict['last_gamma_constraint_loss'].append((global_step, last_gamma_constraint_loss.item()))
                    log_dict['last_gamma_q_values'].append((global_step, old_val[:, -1].mean().item()))
                    log_dict['last_gamma_td_loss'].append((global_step, last_gamma_td_loss.item()))

                    log_dict['first_gamma_td_loss'].append((global_step, first_gamma_td_loss.item()))

                    if args.is_tabular:
                        # I could do the pass through thing instead, maybe that's actually easier.
                        input_states = torch.eye(envs.single_observation_space.shape[0]).to(device)
                        learned_q_values = q_network(input_states).detach().cpu().numpy()
                        # learned_q_values = q_network.network.weight.data.cpu().numpy().reshape(envs.single_observation_space.shape[0], len(gammas), envs.single_action_space.n)
                        # learned_q_values = q_network.network.weight.data.cpu().numpy().reshape(envs.single_observation_space.shape[0], len(gammas), envs.single_action_space.n)
                        assert learned_q_values.shape == true_q_values.shape, f"{learned_q_values.shape} vs {true_q_values.shape}"
                        tabular_total_mse_from_optimal = ((learned_q_values - true_q_values)**2).mean()
                        tabular_smallest_gamma_mse_from_optimal = ((learned_q_values[:,0,:] - true_q_values[:,0,:])**2).mean()
                        tabular_largest_gamma_mse_from_optimal = ((learned_q_values[:,-1,:] - true_q_values[:,-1,:])**2).mean()
                        tabular_total_l1_from_optimal = (np.absolute(learned_q_values - true_q_values)).mean()

                        max_learned_q = learned_q_values.max()
                        max_q_from_buffer = old_val.max()
                        # print(f'tabular error: {tabular_total_mse_from_optimal:.4f}  max q value: {max_learned_q:.4f} max sampled {old_val.max():.4f} actual max q: {true_q_values.max():.4f} td_loss: {td_loss:.4f}')
                        log_dict['tabular_total_mse_from_optimal'].append((global_step, tabular_total_mse_from_optimal))
                        log_dict['tabular_total_l1_from_optimal'].append((global_step, tabular_total_l1_from_optimal))
                        log_dict['tabular_smallest_gamma_mse_from_optimal'].append((global_step, tabular_smallest_gamma_mse_from_optimal))
                        log_dict['tabular_largest_gamma_mse_from_optimal'].append((global_step, tabular_largest_gamma_mse_from_optimal))


                # optimize the model
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        # TODO: my changes here are untested.
        # model_path = f"{write_dir}.cleanrl_model"
        model_path = os.path.join(write_dir, f"{args.exp_name}.cleanrl_model")
        # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model" 
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        import functools
        make_env_atarid = functools.partial(make_env, is_atari=args.is_atari)

        episodic_returns = evaluate(
            model_path,
            make_env_atarid,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=ManyGammaQNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            # push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")
            push_to_hub(args, episodic_returns, repo_id, "DQN", write_dir, f"videos/{run_name}-eval")

    envs.close()
    writer.close()