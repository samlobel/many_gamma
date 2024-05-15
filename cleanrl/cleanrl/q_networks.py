import torch
from torch import nn
import numpy as np

from gamma_utilities import get_constraint_matrix, get_upper_and_lower_bounds
from gradient_based_coefficients import CoefficientsModule

def get_upper_and_lower_bound_pairwise_constraints(q_values, gammas, r_min, r_max):
    # q_values: [bs, num_gammas, num_actions]
    # gammas: [num_gammas]
    # r_min, r_max: float
    # TODO: Should cap it to start so we don't get NaNs.
    # I imagine I should put a little epsilon for the caps to make sure it doesn't explode.
    # Note that there are gamma times more terms here so if we're being fair to the other guys we should divide by gamma.
    # actually if it's mean not sum it should be fine.
    assert len(gammas.shape) == 1, gammas.shape
    EPSILON=1e-6 # Just to make sure we don't divide by zero.
    gammas_reshaped = gammas[None, :, None]
    v_max = r_max / (1 - gammas_reshaped) - EPSILON
    v_min = r_min / (1 - gammas_reshaped) + EPSILON
    capped_q_values = torch.maximum(torch.minimum(q_values, v_max), v_min)
    gamma_times_capped_q_values = gammas_reshaped * capped_q_values
    inner_log_t_frontloaded = (r_max - capped_q_values + gamma_times_capped_q_values) / (r_max - r_min)
    t_front = torch.log(inner_log_t_frontloaded) / torch.log(gammas_reshaped)

    inner_log_t_backloaded = (r_min - capped_q_values + gamma_times_capped_q_values) / (r_min - r_max)
    t_back = torch.log(inner_log_t_backloaded) / torch.log(gammas_reshaped)

    # frontloaded_q_different_gammas. This will have to be [batch_size, num_gammas, num_gammas, num_actions]
    # because we want to compare each gamma to each other gamma.
    # So, should make t_front and t_back that shape
    gammas_repeated = torch.unsqueeze(gammas_reshaped, 1).repeat(1, len(gammas), 1, 1)
    t_front_repeated = torch.unsqueeze(t_front, 2).repeat(1, 1, len(gammas), 1)
    t_back_repeated = torch.unsqueeze(t_back, 2).repeat(1, 1, len(gammas), 1)
    values_front = r_max * (1 - torch.pow(gammas_repeated, t_front_repeated))/(1 - gammas_repeated) + r_min * torch.pow(gammas_repeated, t_front_repeated)/(1 - gammas_repeated)
    values_back  = r_min * (1 - torch.pow(gammas_repeated, t_back_repeated))/(1 - gammas_repeated) + r_max * torch.pow(gammas_repeated, t_back_repeated)/(1 - gammas_repeated)

    # So [0, i, j, 0] means what? It's the ith gamma trying to be consistent with the jth value. But its an estimate of V_i.
    # Which means I think I need to repeat on the other one!!!

    upper_bounds = torch.max(values_front, values_back)
    lower_bounds = torch.min(values_front, values_back)
    # 
    # print('beeng')
    # Okay, I need to figure out whether I'm subtracting rows or columns. Annoying. It's whatever the gammas_repeated says.
    # So, gammas_repeated added a dimension at 1, so I should do the same for q_values.
    q_values_repeated = torch.unsqueeze(q_values, 1).repeat(1, len(gammas), 1, 1)

    violations_above = torch.maximum(q_values_repeated - upper_bounds, torch.tensor(0, dtype=torch.float32))
    violations_below = torch.maximum(lower_bounds - q_values_repeated, torch.tensor(0, dtype=torch.float32))
    violations = torch.maximum(violations_above, violations_below)

    # violations = torch.maximum(q_values_repeated - upper_bounds, lower_bounds - q_values_repeated)
    return upper_bounds, lower_bounds, violations





class ManyGammaQNetwork(nn.Module):
    # TODO: I'm not super happy with the coefficients nonsense. Maybe I should just focus on the pairwise part. Maybe
    # add an option for that being a separate constraint scale or whatever.
    def __init__(self, env, gammas, main_gamma_index=-1, constraint_regularization=0.0, metric="l2", skip_coefficient_solving=False,
                 only_from_lower=False, skip_self_map=False, r_min=-1.0, r_max=1.0,
                 cap_with_vmax=False,
                 vmax_cap_method="pre-coefficient",
                 additive_constant=0.0,
                 additive_multiple_of_vmax=0.0,
                 neural_net_multiplier=1.0,
                 is_tabular=False,
                 use_pairwise_constraints=False,
                 initialization_values=None, # set to tensor if desired, only for tabular.
                ):
        if initialization_values is not None:
            assert is_tabular, "Can only pass something besides none for tabular"
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
        self._vmax_cap_method = vmax_cap_method
        self._additive_constant = additive_constant
        self._additive_multiple_of_vmax = additive_multiple_of_vmax
        self._neural_net_multiplier = neural_net_multiplier
        self._use_pairwise_constraints = use_pairwise_constraints
        self._initialization_values = initialization_values


        if metric == "l2":
            print("For now, transposing! Important change for sure")
            self._constraint_matrix = torch.tensor(
                get_constraint_matrix(gammas, regularization=constraint_regularization,
                                      skip_self_map=skip_self_map, only_from_lower=only_from_lower),
                                      dtype=torch.float32)
            self._constraint_matrix = self._constraint_matrix.T # Sadly necessary until we change the get_constraint_matrix function.
        else:
            coefficient_module = CoefficientsModule(self._gammas, regularization=constraint_regularization, skip_self_map=skip_self_map, only_from_lower=only_from_lower)
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
        # 

        self._minimum_value = self._r_min / (1 - self._gammas)
        self._maximum_value = self._r_max / (1 - self._gammas)

        self.is_tabular = is_tabular

        super().__init__()
        
        self.network = self._make_network(env, len(gammas), is_tabular=is_tabular)

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
        # 
        # print("neat")


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._gammas = self._gammas.to(*args, **kwargs)
        self._constraint_matrix = self._constraint_matrix.to(*args, **kwargs)
        self._upper_bounds = self._upper_bounds.to(*args, **kwargs)
        self._lower_bounds = self._lower_bounds.to(*args, **kwargs)
        self._maximum_value = self._maximum_value.to(*args, **kwargs)
        self._minimum_value = self._minimum_value.to(*args, **kwargs)
        return self

    def _make_network(self, env, num_gammas, is_tabular=False):
        observation_shape = env.single_observation_space.shape
        assert len(observation_shape) in (1, 3), observation_shape
        if is_tabular:
            print("TABULAR TABULAR TABULAR")
            assert len(observation_shape) == 1 # gonna do it this way anyways
            network = nn.Linear(observation_shape[0], num_gammas * env.single_action_space.n) # Nothing fancy to see here.
            # 
            if self._initialization_values is not None:
                assert self._initialization_values.shape[0] == observation_shape[0]
                assert network.weight.data.shape[1] == self._initialization_values.shape[0]
                # new_weights_torch = torch.tensor(self._initialization_values, dtype=torch.float32).view((observation_shape[0], -1)).transpose(0, 1)
                new_weights_torch = self._initialization_values.clone().float().view((observation_shape[0], -1)).transpose(0, 1)
                network.weight.data = nn.Parameter(new_weights_torch)

                # new_weights = torch.tensor(self._optimal_init_values, dtype=torch.float32).reshape((observation_shape[0], -1))
                # new_weights = torch.tensor(self._optimal_init_values, dtype=torch.float32).reshape(network.weight.data.shape)
                # network.weight.data = nn.Parameter(new_weights)
                # network.weight.data = torch.tensor(self._optimal_init_values, dtype=torch.float32).transpose(1, 2).reshape(network.weight.data.shape)
                # network.bias.copy_(torch.zeros_like(network.bias.data))
                network.bias.data = nn.Parameter(torch.zeros_like(network.bias.data))


            return network
            # return nn.Linear(observation_shape[0], num_gammas * env.single_action_space.n) # Nothing fancy to see here.
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
            output = self.network(x / 255.0).view(-1, len(self._gammas), self._num_actions)
        else:
            assert len(x.shape) == 2
            output = self.network(x).view(-1, len(self._gammas), self._num_actions)
        output = output * self._neural_net_multiplier # Either way, also to let 0 do its job. I could say != 1 but why.
        if self._additive_constant:
            output = output + self._additive_constant
        if self._additive_multiple_of_vmax:
            output = output + self._additive_multiple_of_vmax * self._maximum_value[None, :, None]
        
        return output
    
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
        # for i in range(len(output)):
        #     best_action = best_actions[i]
        #     assert torch.allclose(best_values[i, :], output[i, :, best_action])
        # print('nice')

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

    def get_target_value(self, x_next, rewards, dones, output=None, pass_through_constraint=False, cap_by_vmax=False):
        # assert len(x_next.shape) == 4
        assert rewards.shape[0] == x_next.shape[0]
        assert dones.shape[0] == x_next.shape[0]
        assert rewards.shape[1] == 1
        assert dones.shape[1] == 1
        assert len(rewards.shape) == 2
        assert len(dones.shape) == 2
        
        _, best_values = self.get_best_actions_and_values(x_next, output=output)
        if pass_through_constraint:
            # I think this should probably happen on Q. Not sure why it matters SO much but still.
            best_values = self.propagate_and_bound_v(best_values, cap_by_vmax=cap_by_vmax)
        else:
            if cap_by_vmax: # should do this either way to stay true to arguments.
                best_values = torch.maximum(
                    torch.minimum(best_values, self._maximum_value[None, :]),
                    self._minimum_value[None, :])
        # 
        target_values = rewards + self._gammas[None, :] * best_values * (1 - dones)
        return target_values

        # td_target_all_gammas = data.rewards.flatten() + args.gamma * target_max_all_gammas * (1 - data.dones.flatten())

    def get_pairwise_constraints(self, output):
            upper_bounds, lower_bounds, violations = get_upper_and_lower_bound_pairwise_constraints(output, self._gammas, self._r_min, self._r_max)
            return upper_bounds, lower_bounds, violations

    def propagate_and_bound_v(self, output, cap_by_vmax=False):
        assert len(output.shape) == 2
        assert output.shape[1] == len(self._gammas)
        if cap_by_vmax:
            output = torch.maximum(
                torch.minimum(output, self._maximum_value[None, :]),
                self._minimum_value[None, :])

        constraint_computed_output = torch.matmul(output, self._constraint_matrix)# Size [bs, num_gammas]
        assert constraint_computed_output.shape[0] == output.shape[0]
        assert constraint_computed_output.shape[1] == len(self._gammas)
        # Upper is the most that the constraint-computed is allowed to be above the actual.
        # Lower is how much below the actual the constraint-computed is allowed to be.
        # So, constraint_computed is smart averaging. We want to change it as little as possible.
        # So we add/subtract upper/lower from constraint_computed, then clip original. Fine.
        minimum_valid = constraint_computed_output - self._upper_bounds[None, :]
        maximum_valid = constraint_computed_output + self._lower_bounds[None, :]

        clipped_output = torch.maximum(
            torch.minimum(output, maximum_valid),
            minimum_valid)
        # Shouldn't be needed anymore but will do anyways again. Slows it down I think
        if cap_by_vmax:
            clipped_output = torch.maximum(
                torch.minimum(clipped_output, self._maximum_value[None, :]),
                self._minimum_value[None, :])

        amount_changed = ((clipped_output - output)**2).mean().item()
        amount_smallest_gamma_changed = ((clipped_output - output)[:,0]**2).mean().item()
        # print(f"Propagating made total change by: {amount_changed:.7f}")
        # print(f"Propagating made smallest gamma change by: {amount_smallest_gamma_changed:.7f}")
        return clipped_output

    def propagate_and_bound_q(self, output, cap_by_vmax=False):
        # For now, won't do this one.
        raise Exception("Unimplemented")

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

        # 

        output_batch_actions_gammas = output.transpose(1, 2) # Size [bs, num_actions, num_gammas]
        capped_output_batch_actions_gammas = torch.maximum(
            torch.minimum(output_batch_actions_gammas, self._maximum_value[None, None, :]),
            self._minimum_value[None, None, :])
        
        cap_violations = capped_output_batch_actions_gammas - output_batch_actions_gammas # Not sure yet if I should mean or sum.
        cap_violations = cap_violations.transpose(1, 2).detach() # should only be used for logging. # [bs, num_gammas, num_actions]

        if self._cap_with_vmax:
            if self._vmax_cap_method == "pre-coefficient":
                constraint_computed_output = torch.matmul(capped_output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2) # Size [bs, num_gammas, num_actions]
            elif self._vmax_cap_method == "post-coefficient":
                constraint_computed_output = torch.matmul(output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2) # size [bs, num_gammas, num_actions]
                constraint_computed_output = torch.maximum(
                    torch.minimum(constraint_computed_output, self._maximum_value[None, :, None]),
                    self._minimum_value[None, :, None])
            elif self._vmax_cap_method == "separate-regularization":
                constraint_computed_output = torch.matmul(output_batch_actions_gammas, self._constraint_matrix).transpose(1, 2)

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
        # Maybe I should include different losses in here, because I would rather it take place internally than in the training loop.
        # No, don't want to.

        # if self._use_pairwise_constraints:
        # for now let's always compute it, since we want to log both with and without it. Though it might seriously slow things down...
        upper_bounds, lower_bounds, pairwise_violations = self.get_pairwise_constraints(output)

        return {
            'output': output, 
            'constraint_computed_output': constraint_computed_output, # May be from capped things.
            'upper_violations': upper_violations,
            'lower_violations': lower_violations,
            'cap_violations': cap_violations, # abs or square would give a useful statistic.
            # 'pairwise_violations': pairwise_violations if self._use_pairwise_constraints else torch.tensor(0),
            # 'upper_pairwise': upper_bounds if self._use_pairwise_constraints else torch.tensor(0),
            # 'lower_pairwise': lower_bounds if self._use_pairwise_constraints else torch.tensor(0),
            'pairwise_violations': pairwise_violations,
            'upper_pairwise': upper_bounds,
            'lower_pairwise': lower_bounds,
        }




if __name__ == '__main__':
    # Let's do this the right way.
    from gamma_utilities import get_even_spacing
    import gymnasium as gym
    NUM_GAMMAS = 5
    r_min = -1
    r_max = 1
    reward_sequence_1 = [r_max]*15 + [r_min]*300
    reward_sequence_2 = [r_min]*30 + [r_max]*300
    reward_sequence_3 = np.random.uniform(low=r_min, high=r_max, size=(330,)).tolist()
    gammas = get_even_spacing(0.8, 0.99, NUM_GAMMAS)
    values_1 = torch.zeros(len(gammas))
    values_2 = torch.zeros(len(gammas))
    values_3 = torch.zeros(len(gammas))
    for i, gamma in enumerate(gammas):
        values_1[i] = sum([r * gamma**i for i, r in enumerate(reward_sequence_1)])
        values_2[i] = sum([r * gamma**i for i, r in enumerate(reward_sequence_2)])
        values_3[i] = sum([r * gamma**i for i, r in enumerate(reward_sequence_3)])
    values_1 = values_1[None, :, None].repeat(1, 1, 2)
    values_2 = values_2[None, :, None].repeat(1, 1, 2)
    values_3 = values_3[None, :, None].repeat(1, 1, 2)
    def thunk(): return gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([thunk])

    for values in [values_1, values_2, values_3]:
        # for constraint_regularization in [0.001, 0.01, 0.1, 1.0]:
        for constraint_regularization in [0.001, 0.1, 1.0, 0.1, 0.01, 0.001]:
            for skip_self_map in [False, True]:
                for only_from_lower in [True, False]:
                    q_network = ManyGammaQNetwork(envs, gammas=gammas, constraint_regularization=constraint_regularization,
                                                r_min=r_min, r_max=r_max,
                                                skip_self_map=skip_self_map, only_from_lower=only_from_lower)
                    print(q_network._constraint_matrix.shape)
                    print(values.shape)
                    constraint_dict_1 = q_network.get_constraint_computed_values_and_violations(values, output=values)
                    total_violations = constraint_dict_1['upper_violations'].absolute().sum() + constraint_dict_1['lower_violations'].absolute().sum()
                    total_pairwise_violations = constraint_dict_1['pairwise_violations'].absolute().sum()
                    # import ipdb; ipdb.set_trace()
                    # assert torch.allclose(values, constraint_dict_1['constraint_computed_output'])
                    assert total_violations == 0
                    # print(total_pairwise_violations)
                    # assert total_pairwise_violations <= 0.1, total_pairwise_violations
            # import ipdb; ipdb.set_trace()
            # print('neato')
