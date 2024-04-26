import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from comparison_plotting_utils import *
from plotting_utils import get_summary_data, extract_exploration_amounts, load_count_dict, get_true_vs_approx


def get_config(run_name, group_key):
    try:
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)_.*", run_name).group(1)
    except: # if its at the end of the id name
        try:
            return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)$", run_name).group(1)
        except Exception as e:
            print(f"Failed on {run_name}, {group_key}")
            raise e


def default_make_key(log_dir_name, group_keys):
    keys = [get_config(log_dir_name, group_key) for group_key in group_keys]
    key = "_".join(keys)
    return key

def extract_log_dirs_filter(id_to_pkl, group_keys=("rewardscale",), field_to_plot='episodic_return', filter_func=None):
    # Filter only includes ones that are true

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        if filter_func and not filter_func(run_name):
            continue
        try:
            keys = [get_config(run_name, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(run_name, group_keys)
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {run_name}")
            print(e)

    return log_dir_map

def extract_log_dirs(id_to_pkl, group_keys=("rewardscale",), field_to_plot='episodic_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        try:
            keys = [get_config(run_name, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(run_name, group_keys)
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {run_name}")
            print(e)

    return log_dir_map

def extract_log_dirs_group_func(id_to_pkl, group_func=lambda x: x, field_to_plot='episodic_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for run_name, pkl_path in id_to_pkl.items():
        try:
            key = group_func(run_name)
            if key is None:
                continue
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(pkl_path, field_to_plot=field_to_plot)
            log_dir_map[key].append((frames, returns))
        except:
            print(f"Could not extract from {run_name}")

    return log_dir_map


def plot_comparison_learning_curves(
    # id_to_pkl, # dict
    base_dir, #str
    selected_run_names=None,
    # experiment_name=None,
    # stat='eval_episode_lengths',
    group_keys=("constraintlossscale",),
    group_func=None,
    filter_func=None, # Only include things that are "true" in filter. At the moment this is on parsed config name.
    run_name_filter_func=None,
    save_path=None,
    show=True,
    smoothen=10,
    log_dir_path_map=None,
    uniform_truncate=False,
    truncate_max_frames=-1,
    truncate_min_frames=-1,
    ylabel=False,
    legend_loc=None,
    linewidth=2,
    min_seeds=1,
    all_seeds=False,
    title=None,
    min_final_val=None,
    max_final_val=None,
    field_to_plot='episodic_return',
    log_scale=False,
    ):

    # import seaborn as sns
    # NUM_COLORS=100
    # clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    # sns.set_palette(clrs)
    id_to_pkl = gather_pkl_files_from_base_dir(base_dir=base_dir, selected_run_names=selected_run_names)

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"
    if save_path is not None:
        plt.figure(figsize=(24,12))


    # ylabel = ylabel or "Average Return"
    ylabel = ylabel or field_to_plot

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(id_to_pkl=id_to_pkl, group_func=group_func, field_to_plot=field_to_plot)
        else:
            # log_dir_path_map = extract_log_dirs(id_to_pkl=id_to_pkl, group_keys=group_keys, field_to_plot=field_to_plot)
            log_dir_path_map = extract_log_dirs_filter(id_to_pkl=id_to_pkl, group_keys=group_keys, field_to_plot=field_to_plot, filter_func=run_name_filter_func)

    for config in log_dir_path_map:
        if config is None:
            continue
        if filter_func and not filter_func(config):
            continue
        curves = log_dir_path_map[config]
        print(config)
        for curve in curves:
            print(f"\t{len(curve[0])}")
        truncated_xs, truncated_all_ys = truncate_and_interpolate(curves, max_frames=truncate_max_frames, min_frames=truncate_min_frames)
        print(truncated_xs.shape)
        # 
        if len(truncated_all_ys) < min_seeds:
            continue
        if smoothen and smoothen > 0 and len(truncated_xs) < smoothen:
            continue

        if min_final_val is not None:
            # 
            if np.array(truncated_all_ys)[:, -1].mean().item() <= min_final_val:
                print('skipping because min_final_val violated')
                continue
            # else:
            #     print("not skipping")
            #     import ipdb; ipdb.set_trace()
            #     # print("val", np.array(truncated_all_ys)[:, -1].mean().item())
            #     print("vals", np.array(truncated_all_ys)[:, -1].tolist())
        if max_final_val is not None:
            # 
            if np.array(truncated_all_ys)[:, -1].mean().item() >= max_final_val:
                print('skipping because max_final_val violated')
                continue
        # score_array = np.array(truncated_all_ys)
        print(np.max(truncated_all_ys))
        generate_plot(
            # score_array,
            truncated_xs,
            truncated_all_ys,
            label=config,
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds,
            log_scale=log_scale)
    
    # plt.grid()
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if log_scale:
        plt.yscale('log')
    if show:
        if legend_loc:
            plt.legend(loc=legend_loc)
        else:
            plt.legend()
        plt.show()
    
    if save_path is not None:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def get_rmse_for_each_iteration(count_dict):
    exact, approx = get_true_vs_approx(count_dict, "bonus")
    assert len(exact) == len(approx)
    exact = np.asarray(exact)
    approx = np.asarray(approx)
    sq_errors = (exact-approx) ** 2
    root_mean_sq_errors = np.mean(sq_errors) ** 0.5
    return root_mean_sq_errors



if __name__ == "__main__":

    group_func = None
    run_name_filter_func = None
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/dqn_manygamma/first_sweep_fixed"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/dqn_atari_manygamma/first_sweep_fixed"
    # group_keys = (
    #     # "constraintlossscale",
    #     "envid",
    #     # "numgammas",
    #     # "gammalower",
    # )
    # # run_name_filter_func = lambda run_name: "BeamR" in run_name and "numgammas_4" in run_name# Breakout, Pong, BeamRider
    # run_name_filter_func = lambda run_name: "Pong" in run_name # MountainCar, CartPole, Acrobot

    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/classic_control/fixed_big_sweep_1"
    # # run_name_filter_func = lambda run_name: "CartPole" in run_name and "coefficientmetric_abs" in run_name and "frequency_500" in run_name and "lossscale_0.0" not in run_name# CartPole, MountainCar, Acrobot
    # # run_name_filter_func = lambda run_name: "CartPole" in run_name and "coefficientmetric_abs" in run_name and "frequency_500" in run_name and "lossscale_0.0" not in run_name and "gammaspacing_log" in run_name # CartPole, MountainCar, Acrobot
    # # run_name_filter_func = lambda run_name: "Acrobot" in run_name and "coefficientmetric_abs" in run_name and "frequency_500" in run_name and "gammaspacing_log" in run_name # CartPole, MountainCar, Acrobot
    # run_name_filter_func = lambda run_name: "Acrobot" in run_name and "lossscale_0.0" not in run_name and "coefficientmetric_abs" in run_name and "lossscale_10.0" not in run_name and "frequency_100" not in run_name # CartPole, MountainCar, Acrobot
    # # run_name_filter_func = lambda run_name: "Acrobot" in run_name and "coefficientmetric_abs" in run_name # CartPole, MountainCar, Acrobot

    # group_keys = (
    #     "constraintlossscale",
    #     "envid",
    #     # "semigradientconstraint",
    #     # "targetnetworkfrequency",
    #     # "constraintregularization", 
    #     # "gammaspacing",
    #     # "coefficientmetric",
    # )

    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/classic_control/from_below_with_semi_1"
    # run_name_filter_func = lambda run_name: "CartPole" in run_name and "regularization_100.0" in run_name
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/classic_control/vmax_cap_sweep"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/classic_control/cap_with_vmax_1"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/classic_control/vmax_cap_scale_sweep"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/classic_control/cap_with_vmax_and_scale_1"
    # # run_name_filter_func = None
    # run_name_filter_func = lambda run_name: "Acrob" in run_name and "constraintlossscale_0.0_" not in run_name

    # group_keys = (
    #     "envid",
    #     # "numgammas",
    #     "capwithvmax",
    #     # "scaleconstraintlossbyvmax",
    #     # "targetnetworkfrequency",
    #     # "constraintlossscale",
    #     # "constraintregularization",
    #     # "gammaspacing",
    #     # "coefficientmetric",
    # )
    

    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/fixed_reward/zero_reward_first"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/fixed_reward/one_reward_first"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/fixed_reward/first_sweep"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/fixed_reward/first_sweep_both_envs"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/fixed_reward/vmax_cap_method_sweep"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/tabular/ring_4"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/tabular/ring_5"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/tabular/pairwise_ring_1"
    base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/tabular/pairwise_ring_different_inits_fixed"
    base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/tabular/pairwise_ring_2"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/testing_runs/see_whats_wrong"
    
    # constant_env_sweep_1_001__envid_ZeroRewardEnv-v0__constraintregularization_0.01__constraintlossscale_0.0__seed_1__targetnetworkfrequency_5__additiveconstant_0.0__additivemultipleofvmax_0.0__+capwithvmax
    # run_name_filter_func = lambda run_name: "OneRewardEnv" in run_name and "additivemultipleofvmax_0.0" in run_name and "frequency_5_" in run_name
    # run_name_filter_func = lambda run_name: "OneRewardEnv" in run_name and "additiveconstant_0.0" in run_name and "frequency_5_" in run_name
    # run_name_filter_func = lambda run_name: "ZeroRewardEnv" in run_name and "additiveconstant_200.0" in run_name and "additivemultipleofvmax_0.0" in run_name and "frequency_5_" in run_name
    # run_name_filter_func = lambda run_name: "ZeroRewardEnv" in run_name and "additiveconstant_200.0" in run_name and "additivemultipleofvmax_0.0" in run_name and "frequency_5_" in run_name
    # run_name_filter_func = lambda run_name: "ZeroRewardEnv" in run_name and "additiveconstant_0.0" in run_name and "additivemultipleofvmax_2.0" in run_name and "frequency_5_" in run_name

    # run_name_filter_func = lambda run_name: "ZeroRewardEnv" in run_name and "+capwithvmax" in run_name
    # run_name_filter_func = lambda run_name: "NoisyRingTabularEnv-v0" in run_name #and "+capwithvmax" in run_name
    # run_name_filter_func = lambda run_name: "NoisyRingTabularEnv-v0" in run_name and "additivemultipleofvmax_2.0" in run_name
    # run_name_filter_func = lambda run_name: "NoisyRingTabularEnv-v0" in run_name and "additivemultipleofvmax_2.0" in run_name and "additiveconstant_100.0" in run_name and "+capwithvmax" in run_name and "separate-regularization" in run_name
    # run_name_filter_func = lambda run_name: "NoisyRingTabularEnv-v0" in run_name and "additivemultipleofvmax_2.0" in run_name and "additiveconstant_100.0" in run_name and "+capwithvmax" in run_name

    # run_name_filter_func = lambda run_name: "OneRewardEnv" in run_name and (
    #     "-capwithvmax" in run_name or ("+capwithvmax" in run_name and "separate-regularization" in run_name) 
    # and "+capwithvmax" in run_name)

    run_name_filter_func = lambda run_name: "constraintlossscale_0.0" in run_name

    group_keys = (
        "pairwiselossscale",
        "constraintlossscale",
        "additivemultipleofvmax"
    )

    # group_keys = (
    #     # "envid",
    #     # "capwithvmax",
    #     # "targetnetworkfrequency",
    #     # "additiveconstant",
    #     # "additivemultipleofvmax",
    #     "envid",
    #     # "targetnetworkfrequency",
    #     "additiveconstant",
    #     "additivemultipleofvmax",
        
    #     "constraintlossscale",
    #     # "constraintregularization",        
    #     "capwithvmax",
    #     "vmaxcapmethod",
    # )

    # group_keys = (
    #     "constraintregularization",
    #     "constraintlossscale",
    #     "vmaxcapmethod",
    #     "capwithvmax",
    # )


    # run_name_filter_func = lambda run_name: "BeamRider" in run_name and run_name.endswith("constraintlossscale_0.0") # Breakout, Pong, BeamRider
    # run_name_filter_func = lambda run_name: "CartPole" in run_name # MountainCar, CartPole, Acrobot
    # run_name_filter_func = None

    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/ccv_runs/runs/classic_control/compare_semigradient_1/"
    # group_keys = (
    #     "semigradientconstraint",
    #     "constraintlossscale",
    #     # "envid",
    #     # "numgammas",
    #     # "gammalower",
    # )
    # run_name_filter_func = lambda run_name: "CartPole" in run_name # Breakout, Pong, BeamRider
    # run_name_filter_func = lambda run_name: "CartPole" in run_name and "-semi" in run_name

    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/local_runs/classic_control/constraint_regularization_1/"
    # base_dir = "/Users/slobal1/Code/ML/many_gamma/many_gamma/cleanrl/local_runs/classic_control/constraint_regularization_2/"
    # group_keys = (
    #     # "semigradientconstraint",
    #     "constraintlossscale",
    #     # "constraintregularization",
    #     # "targetnetworkfrequency",
    #     # "envid",
    #     # "numgammas",
    #     # "gammalower",
    # )

    # run_name_filter_func = lambda run_name: "-semigradientconstraint" in run_name
    # run_name_filter_func = lambda run_name: "frequency_500" in run_name

    plot_comparison_learning_curves(
        base_dir=base_dir,
        save_path=None,
        show=True,
        # save_path="/Users/slobal1/Downloads/matplotlib_plots/r2d2/visgrid/cfn_tau_1.png",
        # show=False,
        group_keys=group_keys,
        group_func=group_func,
        run_name_filter_func=run_name_filter_func,
        # group_keys=("rewardcoefficient", "spi", "learningrate"),
        # group_keys=("learningrate", ),
        # group_func=rc_group_func,
        smoothen=100,
        # smoothen=False,
        # truncate_min_frames=200000,
        # min_seeds=5,
        # all_seeds=True,
        # title="R2D2 RND sweep",
        # min_final_val=1e4,
        # max_final_val=1e4,
        # field_to_plot="episodic_return", # td_loss, total_loss, SPS, constraint_loss, q_values, episodic_return, episodic_length, 
        # field_to_plot="q_values", # td_loss, total_loss, SPS, constraint_loss, q_values, episodic_return, episodic_length, 
        # field_to_plot="td_loss", # last_gamma_cap_violations_average, last_gamma_constraint_loss, last_gamma_q_values, last_gamma_td_loss
        # field_to_plot="constraint_loss",  # last_gamma_cap_violations_average, last_gamma_constraint_loss, last_gamma_q_values, last_gamma_td_loss
        # field_to_plot="tabular_total_mse_from_optimal",
        # field_to_plot="pairwise_violation_mse",
        field_to_plot="constraint_loss",
        log_scale=True,
        )
