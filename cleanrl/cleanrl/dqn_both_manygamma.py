# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
import pickle
from collections import defaultdict

from gamma_utilities import *
from gradient_based_coefficients import CoefficientsModule

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


@dataclass
class ArgsClassic:
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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    constraint_loss_scale: float = 0.0
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
    r_min: float = -1.0
    """Minimum per-step reward"""
    r_max: float = 1.0
    """Maximum per-step reward"""
    cap_with_vmax: bool = False
    """Whether to cap values with 1/(1-gamma) before inputting to constraint matrix, also keeps constraints below the same."""
    scale_constraint_loss_by_vmax: bool = False
    """Whether to scale each element of constraint loss by 1/(1-gamma)"""

@dataclass
class ArgsAtari:
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
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    constraint_loss_scale: float = 0.0
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
    r_min: float = -1.0
    """Minimum per-step reward"""
    r_max: float = 1.0
    """Maximum per-step reward"""
    cap_with_vmax: bool = False
    """Whether to cap values with 1/(1-gamma) before inputting to constraint matrix, also keeps constraints below the same."""
    scale_constraint_loss_by_vmax: bool = False
    """Whether to scale each element of constraint loss by 1/(1-gamma)"""


def make_env(env_id, seed, idx, capture_video, run_name, is_atari=False):
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

    return thunk_atari if is_atari else thunk_flat


# ALGO LOGIC: initialize agent here:
class ManyGammaQNetwork(nn.Module):
    def __init__(self, env, gammas, main_gamma_index=-1, constraint_regularization=0.0, metric="l2", skip_coefficient_solving=False,
                 only_from_lower=False, r_min=-1.0, r_max=1.0,
                 cap_with_vmax=False):
        assert r_max > r_min
        assert len(gammas) >= 1
        if not isinstance(gammas, (list, tuple)):
            assert len(gammas.shape) == 1
        assert metric in ("l2", "abs"), metric # Whether to use gradient based solver.
        self._observation_space_shape = env.single_observation_space.shape
        self._gammas = torch.tensor(gammas, dtype=torch.float32)
        self._main_gamma_index = main_gamma_index
        self._num_actions = env.single_action_space.n
        self._r_min, self._r_max = r_min, r_max
        self._cap_with_vmax = cap_with_vmax

        if metric == "l2":
            print("For now, transposing! Important change for sure")
            self._constraint_matrix = torch.tensor(get_constraint_matrix(gammas, regularization=constraint_regularization), dtype=torch.float32)
            self._constraint_matrix = self._constraint_matrix.T # Sadly necessary until we change the get_constraint_matrix function.
        else:
            # TODO: principled decision on whether we should zero diagonal or not. Let's go with zero it.
            coefficient_module = CoefficientsModule(self._gammas, regularization=constraint_regularization, skip_self_map=True, only_from_lower=only_from_lower)
            if not skip_coefficient_solving: # We don't care about this for target network, skipping for convenience.
                coefficient_module.solve(num_steps=10001, lr=0.001)
                coefficient_module.solve(num_steps=10001, lr=0.0001)
                coefficient_module.solve(num_steps=10001, lr=0.00001)
                # print("\n\ndont firget\n\n")
                # coefficient_module.solve(num_steps=1001, lr=0.001)
                # coefficient_module.solve(num_steps=1001, lr=0.0001)
                # coefficient_module.solve(num_steps=1001, lr=0.00001)
            self._constraint_matrix = torch.tensor(coefficient_module.get_coefficients(), dtype=torch.float32)

        self._upper_bounds, self._lower_bounds = get_upper_and_lower_bounds(gammas, self._constraint_matrix.T, r_min=r_min, r_max=r_max) # TODO: It's re-transposed here. Make sure thats alright.
        # Upper is the most that the constraint-computed is allowed to be above the actual.
        # Lower is how much below the actual the constraint-computed is allowed to be.
        # Might be that way already.
        # I should asser that the constraint loss is zero when rewards are 1 - gamma.
        self._upper_bounds = torch.tensor(self._upper_bounds, dtype=torch.float32)
        self._lower_bounds = torch.tensor(self._lower_bounds, dtype=torch.float32)
        # import ipdb; ipdb.set_trace()

        self._minimum_value = self._r_min / (1 - self._gammas)
        self._maximum_value = self._r_max / (1 - self._gammas)

        super().__init__()
        
        self.network = self._make_network(env, len(gammas))

        # # Make sure that constraints are not violated for consistent inputs
        # test_output_big = (1 / (1 - self._gammas))[None, ...][..., None].repeat(1, 1, 3)
        # test_output_zero = torch.zeros_like(self._gammas)[None, ...][..., None].repeat(1, 1, 3)
        # test_output_small = -1 * test_output_big
        # constraint_dict_big = self.get_constraint_computed_values_and_violations(test_output_big, output=test_output_big)
        # constraint_dict_zero = self.get_constraint_computed_values_and_violations(test_output_zero, output=test_output_zero)
        # constraint_dict_small = self.get_constraint_computed_values_and_violations(test_output_small, output=test_output_small)
        # assert constraint_dict_big['lower_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # assert constraint_dict_big['upper_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # assert constraint_dict_zero['lower_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # assert constraint_dict_zero['upper_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # assert constraint_dict_small['lower_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # assert constraint_dict_small['upper_violations'].allclose(torch.tensor(0.), atol=1e-5)

        # test_output_violated = 2 * test_output_big
        # constraint_dict_violated = self.get_constraint_computed_values_and_violations(test_output_violated, output=test_output_violated)
        # assert not constraint_dict_violated['lower_violations'].allclose(torch.tensor(0.), atol=1e-5)
        # print('all passed')
        # import ipdb; ipdb.set_trace()
        # print("neat")


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._gammas = self._gammas.to(*args, **kwargs)
        self._constraint_matrix = self._constraint_matrix.to(*args, **kwargs)
        self._upper_bounds = self._upper_bounds.to(*args, **kwargs)
        self._lower_bounds = self._lower_bounds.to(*args, **kwargs)
        return self

    def _make_network(self, env, num_gammas):
        observation_shape = env.single_observation_space.shape
        assert len(observation_shape) in (1, 3)
        if len(observation_shape) == 1:
            print("Flat NN")
            return nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_gammas * env.single_action_space.n),
            )
        else:
            print("Conv NN")
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                # nn.Linear(512, env.single_action_space.n),
                nn.Linear(512, num_gammas * env.single_action_space.n),
            )

    def forward(self, x):
        # Now this returns all the values, reshaped so that each gamma has a vector of action-values
        # Will this always be a batch? I don't really know but let's say yeah.
        # Size [bs, num_gammas, num_actions]
        if len(self._observation_space_shape) == 3:
            assert len(x.shape) == 4
            return self.network(x / 255.0).view(-1, len(self._gammas), self._num_actions)
        else:
            assert len(x.shape) == 2
            return self.network(x).view(-1, len(self._gammas), self._num_actions)
    
    def get_best_actions_and_values(self, x, output=None):
        # Cache output if necessary
        # assert len(x.shape) == 4
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        output = self.forward(x) if output is None else output
        # assert len(output.shape) == len(x.shape) + 1
        best_actions = torch.argmax(output[:, self._main_gamma_index, :], dim=1)
        assert best_actions.shape[0] == x.shape[0]
        assert len(best_actions.shape) == 1
        best_values = output[torch.arange(output.shape[0]), :, best_actions]
        assert best_values.shape[0] == x.shape[0]
        assert best_values.shape[1] == len(self._gammas)
        assert len(best_values.shape) == 2
        return best_actions, best_values

    def get_values_for_action(self, x, actions, output=None):
        assert len(actions.shape) == 1
        assert actions.shape[0] == x.shape[0]
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        # output = self.forward(x)
        output = self.forward(x) if output is None else output
        assert len(output.shape) == 3
        values = output[torch.arange(output.shape[0]), :, actions]
        assert values.shape[0] == x.shape[0]
        assert values.shape[1] == len(self._gammas)
        assert len(values.shape) == 2
        return values

    def get_target_value(self, x, rewards, dones, output=None):
        # assert len(x.shape) == 4
        assert rewards.shape[0] == x.shape[0]
        assert dones.shape[0] == x.shape[0]
        assert rewards.shape[1] == 1
        assert dones.shape[1] == 1
        assert len(rewards.shape) == 2
        assert len(dones.shape) == 2
        
        _, best_values = self.get_best_actions_and_values(x, output=output)
        target_values = rewards + self._gammas[None, :] * best_values * (1 - dones)
        return target_values

        # td_target_all_gammas = data.rewards.flatten() + args.gamma * target_max_all_gammas * (1 - data.dones.flatten())

    def get_constraint_computed_values_and_violations(self, x, semi_gradient=False, normalize=None, output=None):
        # assert normalize in (None, 'none', 'l2', 'l1')
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        # output = self.forward(x)
        output = self.forward(x) if output is None else output # Size [bs, num_gammas, num_actions]

        # # if normalize:
        # if normalize == 'l2':
        #     normalization = output.norm(dim=2, keepdim=True)
        # elif normalize == 'l1':
        #     normalization = output.norm(dim=2, keepdim=True, p=1)
        # elif normalize == 'none' or normalize is None:
        #     normalization = 1
        # else:
        #     print(normalize)
        #     raise Exception("Shouldn't be here")

        # output = output / normalization
        
        # if normalize == 'l2':
        #     output = output / output.norm(dim=2, keepdim=True)
        # elif normalize == 'l1':
        #     output = output / output.norm(dim=2, keepdim=True, p=1)
        # elif normalize == 'none' or normalize is None:
        #     pass
        # else:
        #     print(normalize)
        #     raise Exception("Shouldn't be here")

        # import ipdb; ipdb.set_trace()
        output_batch_actions_gammas = output.transpose(1, 2) # Size [bs, num_actions, num_gammas]
        capped_output_batch_actions_gammas = torch.maximum(
            torch.minimum(output_batch_actions_gammas, self._maximum_value[None, None, :]),
            self._minimum_value[None, None, :])
        
        cap_violations = capped_output_batch_actions_gammas - output_batch_actions_gammas # Not sure yet if I should mean or sum.
        cap_violations = cap_violations.transpose(1, 2).detach() # should only be used for logging. # [bs, num_gammas, num_actions]

        if self._cap_with_vmax:
            constraint_computed_output = torch.matmul(capped_output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2) # Size [bs, num_gammas, num_actions]
        else:
            constraint_computed_output = torch.matmul(output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2) # Size [bs, num_gammas, num_actions]




        # constraint_computed_output = torch.matmul(output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2) # Size [bs, num_gammas, num_actions]
        assert constraint_computed_output.shape[0] == x.shape[0]
        assert constraint_computed_output.shape[1] == len(self._gammas)
        assert constraint_computed_output.shape[2] == self._num_actions

        if semi_gradient:
            # In this case we only want to send the outputs towards the constraints.
            # I think this makes more sense.
            constraint_computed_output = constraint_computed_output.detach()

        difference = constraint_computed_output - output
        # difference = output -constraint_computed_output
        ub = self._upper_bounds[None, :, None] # Sounds like 
        lb = self._lower_bounds[None, :, None]

        ## This would be a less confusing way to write that out.
        # max_upper = constraint_computed_output + ub
        # min_lower = constraint_computed_output - lb
        # new_upper_violation = torch.relu(output - max_upper)
        # new_lower_violation = torch.relu(min_lower - output)

        # ub = ub / normalization
        # lb = lb / normalization
        upper_violations = torch.maximum(difference - ub, torch.tensor(0, dtype=torch.float32))
        # lower_violations = torch.maximum(lb - difference, torch.tensor(0, dtype=torch.float32))
        # if lb is 3 and diff is positive, it should be 0.
        # If lb is 3 and diff is -2, it should still be 0
        # If lb is 3 and diff is -4, it should be 1. -lb - diff. Nice.
        # Pretty much, the way I phrase my constraints was super confusing 
        lower_violations = torch.maximum(-lb - difference, torch.tensor(0, dtype=torch.float32))
        return {
            'output': output, 
            'constraint_computed_output': constraint_computed_output, # May be from capped things.
            'upper_violations': upper_violations,
            'lower_violations': lower_violations,
            'cap_violations': cap_violations, # abs or square would give a useful statistic.
        }



    # def get_score_estimates(self, x, actions):



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
    if args.is_atari:
        args = tyro.cli(ArgsAtari) # Different defaults

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.tag:
        run_name = args.tag
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.tag + '__' if args.tag else ''}{int(time.time())}"

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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, is_atari=args.is_atari) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    assert args.gamma_spacing in ("even", "log", "linear"), args.gamma_spacing
    gamma_choosing_func = {"even": get_even_spacing, "log": get_even_log_spacing, "linear": np.linspace}[args.gamma_spacing]
    gammas = gamma_choosing_func(args.gamma_lower, args.gamma_upper, args.num_gammas)
    print(gammas)

    # So, it doesn't transfer coefficients etc.
    q_network = ManyGammaQNetwork(
        envs, gammas, constraint_regularization=args.constraint_regularization,
        metric=args.coefficient_metric, only_from_lower=args.only_from_lower,
        r_min=args.r_min, r_max=args.r_max,
        cap_with_vmax=args.cap_with_vmax).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = ManyGammaQNetwork(
        envs, gammas, constraint_regularization=args.constraint_regularization,
        metric=args.coefficient_metric, only_from_lower=args.only_from_lower,
        r_min=args.r_min, r_max=args.r_max,
        cap_with_vmax=args.cap_with_vmax).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        # TODO: Have methods take in the result of forward, so I don't have to recompute, should save me ~25% compute.
        # TODO: add DDQN feature, in case helpful.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # Not max for all gammas, but action is max action for main gamma.
                    # target_max_all_gammas = q_network.get_best_actions_and_values(torch.Tensor(obs).to(device))[1].cpu().numpy()
                    # target_max, _ = target_network(data.next_observations).max(dim=1)
                    # td_target_all_gammas = data.rewards.flatten() + args.gamma * target_max_all_gammas * (1 - data.dones.flatten())
                    td_target = target_network.get_target_value(data.next_observations, data.rewards, data.dones)
                    # raise Exception("Here")
                    # td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                # old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # Cache it!
                q_outputs = q_network(data.observations)
                # old_val = q_network.get_values_for_action(data.observations, data.actions.squeeze())
                old_val = q_network.get_values_for_action(data.observations, data.actions.squeeze(), output=q_outputs) # [bs, num_gammas]
                # raise Exception("here")
                td_loss = F.mse_loss(td_target, old_val)
                last_gamma_td_loss = F.mse_loss(td_target[:, -1], old_val[:, -1]) # was bs x n_gammas (no actions anymore)

                violation_dict = q_network.get_constraint_computed_values_and_violations(
                    data.observations, semi_gradient=args.semigradient_constraint,
                    normalize=args.constraint_normalization, output=q_outputs)
                upper_violations = violation_dict['upper_violations'] # actions last
                lower_violations = violation_dict['lower_violations'] # actions last
                cap_violations_average = (violation_dict['cap_violations'] ** 2).mean()
                last_gamma_cap_violations_average = (violation_dict['cap_violations'][:, -1, :] ** 2).mean()
                last_gamma_constraint_loss = 0.5*((upper_violations[:, -1, :]**2).mean() + (lower_violations[:, -1, :]**2).mean())
                # import ipdb; ipdb.set_trace()
                # td_loss = loss
                # constraint_loss = (upper_violations.mean() + lower_violations.mean())
                constraint_loss = 0.5*((upper_violations**2).mean() + (lower_violations**2).mean())

                scaled_upper_violations = upper_violations * (1 - q_network._gammas[None, :, None])
                scaled_lower_violations = lower_violations * (1 - q_network._gammas[None, :, None])
                scaled_constraint_loss = 0.5*((scaled_upper_violations**2).mean() + (scaled_lower_violations**2).mean())

                if args.constraint_loss_scale > 0:
                    if args.scale_constraint_loss_by_vmax:
                        total_loss = td_loss + (args.constraint_loss_scale * scaled_constraint_loss)
                    else:
                        total_loss = td_loss + (args.constraint_loss_scale * constraint_loss)
                else:
                    total_loss = td_loss

                log_frequency = 1000 if args.is_atari else 100
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
                    log_dict['q_values'].append((global_step, old_val.mean().item()))
                    log_dict['SPS'].append((global_step, int(global_step / (time.time() - start_time))))
                    log_dict['cap_violations_average'].append((global_step, cap_violations_average.item()))

                    log_dict['last_gamma_cap_violations_average'].append((global_step, last_gamma_cap_violations_average.item()))
                    log_dict['last_gamma_constraint_loss'].append((global_step, last_gamma_constraint_loss.item()))
                    log_dict['last_gamma_q_values'].append((global_step, old_val[:, -1].mean().item()))
                    log_dict['last_gamma_td_loss'].append((global_step, last_gamma_td_loss.item()))

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