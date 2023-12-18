# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
import pickle
from collections import defaultdict

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
class Args:
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
    gamma_upper: float = 0.99
    num_gammas: int = 2
    tag: str = ""
    log_dir: str = "runs"

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
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

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)



from scipy import linalg

def get_difference(known_gammas, coefficients, gamma_target, final_step = 1000):
    # Numerically integrates upper and lower difference bounds.
    # Pos diff is the most that combined could possibly be above target.
    # Neg diff is the most that combined could possibly be below target.
    def output(x):
        v_combined = 0
        for c, g in zip(coefficients, known_gammas):
            v_combined += c * (g ** x)
        return v_combined
    x_axis = np.linspace(0, final_step, final_step + 1)
    combined_output = output(x_axis)
    target_output = gamma_target ** x_axis
    diff = combined_output - target_output
    allowed_above = np.maximum(diff, 0).sum()
    allowed_below = -1 * np.minimum(diff, 0).sum()
    assert allowed_below >= 0
    assert allowed_above >= 0
    return allowed_above, allowed_below


def compute_coefficients(known_gammas: tuple, gamma_target: float):
    A = []
    b = []
    for gamma_i in known_gammas:
        A_i = [1 / (1 - gamma_i * gamma_j) for gamma_j in known_gammas]
        b_i = 1 / (1 - gamma_i * gamma_target)
        A.append(A_i)
        b.append(b_i)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    coefficients = linalg.solve(A, b).tolist()
    # print(coefficients)
    return coefficients

def get_constraint_matrix(gammas):
    gammas = np.array(gammas).tolist() # make it a list
    matrix = []
    for i in range(len(gammas)):
        this_gamma = gammas[i]
        other_gammas = gammas[:i] + gammas[i+1:]
        coefficients = compute_coefficients(other_gammas, this_gamma)


        coefficients = coefficients[:i] + [0] + coefficients[i:]
        matrix.append(coefficients)

    matrix = np.array(matrix)
    assert np.max(matrix * np.eye(len(gammas))) == 0, "all should be zero"
    return matrix

def get_upper_and_lower_bounds(gammas, coefficient_matrix, final_step=10000):
    allowed_aboves = []
    allowed_belows = []
    for i, coefficient_list in enumerate(coefficient_matrix):
        gamma = gammas[i]
        allowed_above, allowed_below = get_difference(gammas, coefficient_list, gamma, final_step=final_step)
        allowed_aboves.append(allowed_above)
        allowed_belows.append(allowed_below)

    return np.array(allowed_aboves), np.array(allowed_belows)

def get_even_spacing(gamma_small, gamma_big, total_points):
    assert gamma_small < gamma_big
    assert total_points >= 2
    vmax_small = 1 / (1 - gamma_small)
    vmax_big = 1 / (1 - gamma_big)
    spaces = np.linspace(vmax_small, vmax_big, total_points)
    inverted = 1 - 1 / spaces
    return inverted.tolist()

class ManyGammaQNetwork(nn.Module):
    def __init__(self, env, gammas, main_gamma_index=-1):
        assert len(gammas) >= 1
        if not isinstance(gammas, (list, tuple)):
            assert len(gammas.shape) == 1
        self._gammas = torch.tensor(gammas, dtype=torch.float32)
        self._main_gamma_index = main_gamma_index
        self._num_actions = env.single_action_space.n
        self._constraint_matrix = torch.tensor(get_constraint_matrix(gammas), dtype=torch.float32)
        self._upper_bounds, self._lower_bounds = get_upper_and_lower_bounds(gammas, self._constraint_matrix)
        self._upper_bounds = torch.tensor(self._upper_bounds, dtype=torch.float32)
        self._lower_bounds = torch.tensor(self._lower_bounds, dtype=torch.float32)


        super().__init__()
        # class Thing(nn.Module):
        #     def forward(self, x):
        #         return x * 0
        
        self.network = nn.Sequential(
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
            nn.Linear(512, len(gammas) * env.single_action_space.n),
        )

        # self.network = nn.Sequential(
        #     nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, len(gammas) * env.single_action_space.n),
        #     # Thing(),
        # )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._gammas = self._gammas.to(*args, **kwargs)
        self._constraint_matrix = self._constraint_matrix.to(*args, **kwargs)
        self._upper_bounds = self._upper_bounds.to(*args, **kwargs)
        self._lower_bounds = self._lower_bounds.to(*args, **kwargs)
        return self


    # def forward(self, x):
    #     return self.network(x / 255.0)
    def forward(self, x):
        # Now this returns all the values, reshaped so that each gamma has a vector of action-values
        # Will this always be a batch? I don't really know but let's say yeah.
        assert len(x.shape) == 4 # I think.
        return self.network(x / 255.0).view(-1, len(self._gammas), self._num_actions)
    
    def get_best_actions_and_values(self, x):
        assert len(x.shape) == 4
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        output = self.forward(x)
        assert len(output.shape) == 3
        best_actions = torch.argmax(output[:, self._main_gamma_index, :], dim=1)
        assert best_actions.shape[0] == x.shape[0]
        assert len(best_actions.shape) == 1
        best_values = output[torch.arange(output.shape[0]), :, best_actions]
        assert best_values.shape[0] == x.shape[0]
        assert best_values.shape[1] == len(self._gammas)
        assert len(best_values.shape) == 2
        return best_actions, best_values

    def get_values_for_action(self, x, actions):
        assert len(x.shape) == 4
        assert len(actions.shape) == 1
        assert actions.shape[0] == x.shape[0]
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        output = self.forward(x)
        assert len(output.shape) == 3
        values = output[torch.arange(output.shape[0]), :, actions]
        assert values.shape[0] == x.shape[0]
        assert values.shape[1] == len(self._gammas)
        assert len(values.shape) == 2
        return values

    def get_target_value(self, x, rewards, dones):
        assert len(x.shape) == 4
        assert rewards.shape[0] == x.shape[0]
        assert dones.shape[0] == x.shape[0]
        assert rewards.shape[1] == 1
        assert dones.shape[1] == 1
        assert len(rewards.shape) == 2
        assert len(dones.shape) == 2
        
        _, best_values = self.get_best_actions_and_values(x)
        target_values = rewards + self._gammas[None, :] * best_values * (1 - dones)
        return target_values

        # td_target_all_gammas = data.rewards.flatten() + args.gamma * target_max_all_gammas * (1 - data.dones.flatten())

    def get_constraint_computed_values_and_violations(self, x):
        # output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        output = self.forward(x)

        # import ipdb; ipdb.set_trace()
        constraint_computed_output = torch.matmul(output.transpose(1, 2), self._constraint_matrix).transpose(1, 2)
        assert constraint_computed_output.shape[0] == x.shape[0]
        assert constraint_computed_output.shape[1] == len(self._gammas)
        assert constraint_computed_output.shape[2] == self._num_actions

        difference = constraint_computed_output - output
        # difference = output -constraint_computed_output
        ub = self._upper_bounds[None, :, None] # Sounds like 
        lb = self._lower_bounds[None, :, None]
        upper_violations = torch.maximum(difference - ub, torch.tensor(0, dtype=torch.float32))
        # lower_violations = torch.maximum(lb - difference, torch.tensor(0, dtype=torch.float32))
        # if lb is 3 and diff is positive, it should be 0.
        # If lb is 3 and diff is -2, it should still be 0
        # If lb is 3 and diff is -4, it should be 1. -lb - diff. Nice.
        # Pretty much, the way I phrase my constraints was super confusing 
        lower_violations = torch.maximum(-lb - difference, torch.tensor(0, dtype=torch.float32))
        return {
            'output': output, 
            'constraint_computed_output': constraint_computed_output,
            'upper_violations': upper_violations,
            'lower_violations': lower_violations,
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
    args = tyro.cli(Args)
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # q_network = QNetwork(envs).to(device)
    gammas = get_even_spacing(args.gamma_lower, args.gamma_upper, args.num_gammas)
    q_network = ManyGammaQNetwork(envs, gammas).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # target_network = QNetwork(envs).to(device)
    target_network = ManyGammaQNetwork(envs, gammas).to(device)
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
                old_val = q_network.get_values_for_action(data.observations, data.actions.squeeze())
                # raise Exception("here")
                td_loss = F.mse_loss(td_target, old_val)

                violation_dict = q_network.get_constraint_computed_values_and_violations(data.observations)
                upper_violations = violation_dict['upper_violations']
                lower_violations = violation_dict['lower_violations']
                # import ipdb; ipdb.set_trace()
                # td_loss = loss
                # constraint_loss = (upper_violations.mean() + lower_violations.mean())
                constraint_loss = 0.5*((upper_violations**2).mean() + (lower_violations**2).mean())
                if args.constraint_loss_scale > 0:
                    total_loss = td_loss + args.constraint_loss_scale * constraint_loss
                else:
                    total_loss = td_loss

                if global_step % 100 == 0:
                    # writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/td_loss", td_loss, global_step)
                    writer.add_scalar("losses/total_loss", total_loss, global_step)
                    # if args.constraint_loss_scale > 0:
                    writer.add_scalar("losses/constraint_loss", constraint_loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    log_dict['td_loss'].append((global_step, td_loss.item()))
                    log_dict['total_loss'].append((global_step, total_loss.item()))
                    log_dict['constraint_loss'].append((global_step, constraint_loss.item()))
                    log_dict['q_values'].append((global_step, old_val.mean().item()))
                    log_dict['SPS'].append((global_step, int(global_step / (time.time() - start_time))))

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

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
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